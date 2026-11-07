"""Typed configuration, seed loading, and the pre-registration lock for thresholds.

The lock stores a content hash of the *parsed* thresholds (so line endings / comments do not
matter). The test split may only be scored when the current thresholds match the registered hash.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"
LOCK_PATH = CONFIG_DIR / "thresholds.lock.json"


class PreregistrationError(RuntimeError):
    """Raised when test-set evaluation is attempted without a valid registration."""


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_seeds(path: Optional[Path] = None) -> Dict[str, int]:
    """Integer seeds from configs/seeds.yaml - the single source of randomness."""
    return {k: int(v) for k, v in _load_yaml(path or CONFIG_DIR / "seeds.yaml").items()}


def load_models_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _load_yaml(path or CONFIG_DIR / "models.yaml")


class Gate1(BaseModel):
    target_auroc_min: float
    target_ece_max: float
    target_brier_max: float
    nondeterminism_divergence_max: float


class Phase1Protocol(BaseModel):
    calibrators: List[str] = Field(default_factory=lambda: ["temperature", "isotonic"])
    selection_rule: str = "lowest_validation_ece"
    n_bins: int = 10
    repeats: int = 1
    bootstrap_iterations: int = 1000
    ci_level: float = 0.95
    cluster_unit: str = "source_document"
    pass_rule: str = "point_estimate_meets_all_gate1_targets"
    nondeterminism_probe_n: int = 100
    wording_probe_n: int = 100


class Thresholds(BaseModel):
    acceptance_threshold: float
    abstention_threshold: float
    clip_bounds: Tuple[float, float] = (0.01, 0.99)
    gate1: Gate1
    gate2: Dict[str, Any] = Field(default_factory=dict)
    fusion_features: List[str] = Field(default_factory=list)
    phase1: Phase1Protocol = Field(default_factory=Phase1Protocol)


def load_thresholds(path: Optional[Path] = None) -> Thresholds:
    return Thresholds(**_load_yaml(path or THRESHOLDS_PATH))


def thresholds_hash(path: Optional[Path] = None) -> str:
    return canonical_hash(_load_yaml(path or THRESHOLDS_PATH))


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def registration_is_committed(thresholds_path: Path = THRESHOLDS_PATH, lock_path: Path = LOCK_PATH) -> bool:
    """True only if both files are tracked by git with no uncommitted changes."""
    for p in (thresholds_path, lock_path):
        try:
            rel = str(p.relative_to(REPO_ROOT)) if p.is_absolute() else str(p)
        except ValueError:  # outside the repository: cannot be committed to it
            return False
        if _git("ls-files", "--error-unmatch", rel) is None:
            return False
        if _git("status", "--porcelain", "--", rel):
            return False
    return True


def register_thresholds(
    thresholds_path: Path = THRESHOLDS_PATH, lock_path: Path = LOCK_PATH, force: bool = False
) -> Dict[str, Any]:
    """Write the registration lock for the current thresholds (refuses to silently overwrite)."""
    digest = thresholds_hash(thresholds_path)
    if lock_path.exists() and not force:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("sha256") == digest:
            return existing
        raise PreregistrationError(
            "A registration already exists for different thresholds. Re-registering defeats "
            "pre-registration; pass force=True only if no test result has been seen."
        )
    lock = {"sha256": digest, "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def verify_registration(
    thresholds_path: Path = THRESHOLDS_PATH, lock_path: Path = LOCK_PATH
) -> Dict[str, Any]:
    """Return registration info, or raise PreregistrationError if absent/stale."""
    if not lock_path.exists():
        raise PreregistrationError("Thresholds are not registered. Run: python -m src.runs register")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("sha256") != thresholds_hash(thresholds_path):
        raise PreregistrationError(
            "thresholds.yaml changed since registration; refusing to score the test split."
        )
    return {**lock, "committed": registration_is_committed(thresholds_path, lock_path)}
