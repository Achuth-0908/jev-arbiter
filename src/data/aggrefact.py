"""LLM-AggreFact loader (real data), document-grouped splitting, and seeded subsampling.

Dataset: https://huggingface.co/datasets/lytang/LLM-AggreFact (gated, CC-BY-ND-4.0).
The parquet files are downloaded into ``data/aggrefact/`` (gitignored) using ``HF_TOKEN``
from the environment. The token is never logged or written anywhere.

Protocol (see AGENTS.md section 3):
  * official ``test`` split is the test set and is never used for tuning;
  * official ``dev`` split is divided into calibration/validation **by source document**
    so no document ever appears in more than one split.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from ..config import load_seeds
from .loaders import FactClaimInstance

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "aggrefact"
_HF_BASE = "https://huggingface.co/datasets/lytang/LLM-AggreFact/resolve/main/data"
_FILES = {"dev": "dev-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


class DatasetUnavailableError(RuntimeError):
    """Raised when AggreFact cannot be loaded (missing token / terms not accepted)."""


def doc_group_id(doc: str) -> str:
    """Stable identifier of a source document; the unit that must not cross splits."""
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]


def ensure_downloaded(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Return the local parquet path for ``split``, downloading it once if absent."""
    if split not in _FILES:
        raise ValueError(f"split must be one of {sorted(_FILES)}, got {split!r}")
    target = Path(data_dir) / f"{split}.parquet"
    if target.exists():
        return target

    token = os.getenv("HF_TOKEN", "")
    if not token:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            token = os.getenv("HF_TOKEN", "")
        except ImportError:  # pragma: no cover
            pass
    if not token:
        raise DatasetUnavailableError(
            "AggreFact not found locally and HF_TOKEN is not set. Accept the dataset terms on "
            "Hugging Face and add HF_TOKEN to .env."
        )

    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(
        f"{_HF_BASE}/{_FILES[split]}",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=True,
        timeout=300,
    )
    if resp.status_code in (401, 403):
        raise DatasetUnavailableError(
            f"HTTP {resp.status_code} fetching AggreFact: accept the dataset terms for the account "
            "that owns HF_TOKEN."
        )
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def load_aggrefact(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    subsets: Optional[Sequence[str]] = None,
) -> List[FactClaimInstance]:
    """Load an official AggreFact split as ``FactClaimInstance`` objects.

    ``label`` is 1 = supported, 0 = unsupported (as provided by the dataset).
    ``metadata['group']`` is the source-document id used for leakage-safe splitting.
    """
    import pandas as pd

    df = pd.read_parquet(ensure_downloaded(split, data_dir))
    if subsets is not None:
        df = df[df["dataset"].isin(list(subsets))]

    instances: List[FactClaimInstance] = []
    for row in df.itertuples(index=False):
        gid = doc_group_id(row.doc)
        cid = hashlib.sha256(f"{gid}|{row.claim}".encode("utf-8")).hexdigest()[:16]
        instances.append(
            FactClaimInstance(
                id=f"{split}:{row.dataset}:{cid}",
                claim=row.claim,
                evidence=row.doc,
                label=int(row.label),
                subdataset=str(row.dataset),
                metadata={"group": gid, "split_source": split},
            )
        )
    return instances


def create_grouped_splits(
    instances: Sequence[FactClaimInstance],
    ratios: Tuple[float, float, float] = (0.40, 0.20, 0.40),
    seed: Optional[int] = None,
    group_fn: Optional[Callable[[FactClaimInstance], str]] = None,
) -> Tuple[List[FactClaimInstance], List[FactClaimInstance], List[FactClaimInstance]]:
    """Split into (cal, val, test) so that every group lands in exactly one split.

    Groups (default: source document) are shuffled with ``seed`` (default ``split_seed`` from
    configs/seeds.yaml) and assigned within each sub-dataset to approximate ``ratios`` by
    instance count. Prevents the document-level leakage that per-claim splitting causes.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    if seed is None:
        seed = load_seeds()["split_seed"]
    group_fn = group_fn or (lambda inst: str(inst.metadata.get("group") or doc_group_id(inst.evidence)))

    rng = np.random.RandomState(seed)
    by_sub: Dict[str, Dict[str, List[FactClaimInstance]]] = {}
    for inst in instances:
        by_sub.setdefault(inst.subdataset, {}).setdefault(group_fn(inst), []).append(inst)

    out: Tuple[List[FactClaimInstance], ...] = ([], [], [])
    for sub in sorted(by_sub):
        groups = by_sub[sub]
        keys = sorted(groups)  # sort first so shuffling is independent of input order
        order = rng.permutation(len(keys))
        total = sum(len(g) for g in groups.values())
        targets = [r * total for r in ratios]
        filled = [0.0, 0.0, 0.0]
        for idx in order:
            members = groups[keys[idx]]
            # assign to the split furthest below its target share
            deficits = [
                (targets[i] - filled[i]) / targets[i] if targets[i] > 0 else -np.inf
                for i in range(3)
            ]
            dest = int(np.argmax(deficits))
            out[dest].extend(members)
            filled[dest] += len(members)
    return out[0], out[1], out[2]


def stratified_subsample(
    instances: Sequence[FactClaimInstance],
    max_per_subdataset: int,
    seed: Optional[int] = None,
) -> List[FactClaimInstance]:
    """Cap each sub-dataset at ``max_per_subdataset`` (seeded, order-independent).

    Stops the largest sub-dataset (RAGTruth, >50% of AggreFact) from dominating metrics and cost.
    """
    if seed is None:
        seed = load_seeds()["sampling_seed"]
    rng = np.random.RandomState(seed)
    by_sub: Dict[str, List[FactClaimInstance]] = {}
    for inst in instances:
        by_sub.setdefault(inst.subdataset, []).append(inst)

    picked: List[FactClaimInstance] = []
    for sub in sorted(by_sub):
        pool = sorted(by_sub[sub], key=lambda i: i.id)
        if len(pool) > max_per_subdataset:
            idx = rng.choice(len(pool), size=max_per_subdataset, replace=False)
            pool = [pool[i] for i in sorted(idx)]
        picked.extend(pool)
    return picked


def build_protocol_splits(
    max_per_subdataset_dev: int = 200,
    max_per_subdataset_test: int = 200,
    data_dir: Path = DEFAULT_DATA_DIR,
    subsets: Optional[Sequence[str]] = None,
) -> Tuple[List[FactClaimInstance], List[FactClaimInstance], List[FactClaimInstance]]:
    """Calibration/validation from official dev (grouped by document); test = official test."""
    dev_all = load_aggrefact("dev", data_dir, subsets)
    test_all = load_aggrefact("test", data_dir, subsets)
    # The official dev/test files share some source documents (e.g. ~36% of AggreFact-CNN test
    # claims). Test is sacrosanct, so drop those documents from the calibration pool.
    test_groups = {str(i.metadata["group"]) for i in test_all}
    dev_all = [i for i in dev_all if str(i.metadata["group"]) not in test_groups]
    dev = stratified_subsample(dev_all, max_per_subdataset_dev)
    test = stratified_subsample(test_all, max_per_subdataset_test)
    cal, val, _ = create_grouped_splits(dev, ratios=(2 / 3, 1 / 3, 0.0))
    return cal, val, test


def assert_no_group_overlap(*splits: Sequence[FactClaimInstance]) -> None:
    """Raise if any source document appears in more than one split."""
    seen: Dict[str, int] = {}
    for i, split in enumerate(splits):
        for inst in split:
            gid = str(inst.metadata.get("group") or doc_group_id(inst.evidence))
            if seen.setdefault(gid, i) != i:
                raise AssertionError(f"document {gid} appears in splits {seen[gid]} and {i}")
