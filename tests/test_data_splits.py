"""Leakage and determinism tests for document-grouped splitting (offline, synthetic input)."""

import numpy as np
import pytest

from src.data.aggrefact import (
    DEFAULT_DATA_DIR,
    assert_no_group_overlap,
    build_protocol_splits,
    create_grouped_splits,
    doc_group_id,
    stratified_subsample,
)
from src.data.loaders import FactClaimInstance


def _make(n_docs: int = 60, claims_per_doc: int = 5, seed: int = 0):
    rng = np.random.RandomState(seed)
    out = []
    for d in range(n_docs):
        doc = f"document number {d} " * 5
        sub = ["A", "B", "C"][d % 3]
        for c in range(claims_per_doc):
            out.append(
                FactClaimInstance(
                    id=f"{d}-{c}", claim=f"claim {d}.{c}", evidence=doc,
                    label=int(rng.rand() < 0.6), subdataset=sub,
                    metadata={"group": doc_group_id(doc)},
                )
            )
    return out


def test_no_document_crosses_splits():
    cal, val, test = create_grouped_splits(_make(), seed=42)
    assert len(cal) and len(val) and len(test)
    assert_no_group_overlap(cal, val, test)


def test_old_claim_level_split_would_leak():
    """Documents the bug the grouped splitter fixes: per-claim splitting leaks documents."""
    from src.data.loaders import create_frozen_splits

    cal, val, test = create_frozen_splits(_make(), seed=42)
    with pytest.raises(AssertionError):
        assert_no_group_overlap(cal, val, test)


def test_split_is_deterministic_and_order_independent():
    data = _make()
    a = create_grouped_splits(data, seed=42)
    b = create_grouped_splits(list(reversed(data)), seed=42)
    assert [sorted(i.id for i in s) for s in a] == [sorted(i.id for i in s) for s in b]


def test_different_seed_changes_assignment():
    data = _make()
    a = create_grouped_splits(data, seed=1)
    b = create_grouped_splits(data, seed=2)
    assert sorted(i.id for i in a[2]) != sorted(i.id for i in b[2])


def test_ratios_approximately_respected():
    data = _make(n_docs=300)
    cal, val, test = create_grouped_splits(data, ratios=(0.4, 0.2, 0.4), seed=42)
    n = len(data)
    assert abs(len(cal) / n - 0.4) < 0.05
    assert abs(len(val) / n - 0.2) < 0.05
    assert abs(len(test) / n - 0.4) < 0.05


def test_zero_ratio_split_stays_empty():
    cal, val, test = create_grouped_splits(_make(), ratios=(2 / 3, 1 / 3, 0.0), seed=42)
    assert test == [] and cal and val


def test_subsample_caps_each_subdataset_deterministically():
    data = _make(n_docs=90)
    s1 = stratified_subsample(data, max_per_subdataset=20, seed=7)
    s2 = stratified_subsample(list(reversed(data)), max_per_subdataset=20, seed=7)
    assert [i.id for i in s1] == [i.id for i in s2]
    for sub in ("A", "B", "C"):
        assert sum(i.subdataset == sub for i in s1) == 20


@pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "dev.parquet").exists() or not (DEFAULT_DATA_DIR / "test.parquet").exists(),
    reason="AggreFact parquet not downloaded (needs HF_TOKEN); offline CI skips this",
)
def test_real_protocol_splits_have_no_leakage():
    cal, val, test = build_protocol_splits(max_per_subdataset_dev=60, max_per_subdataset_test=60)
    assert cal and val and test
    assert_no_group_overlap(cal, val, test)
    # test split must come only from the official test file, never from dev
    assert all(i.id.startswith("test:") for i in test)
    assert all(i.id.startswith("dev:") for i in cal + val)
