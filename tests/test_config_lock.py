"""Pre-registration lock: the test split must be unreachable unless thresholds match the lock."""

import json

import pytest
import yaml

from src.config import (
    PreregistrationError,
    canonical_hash,
    load_thresholds,
    register_thresholds,
    thresholds_hash,
    verify_registration,
)


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def paths(tmp_path):
    thr = tmp_path / "thresholds.yaml"
    _write(thr, {"acceptance_threshold": 0.7, "gate1": {"target_auroc_min": 0.75}})
    return thr, tmp_path / "thresholds.lock.json"


def test_unregistered_is_refused(paths):
    thr, lock = paths
    with pytest.raises(PreregistrationError, match="not registered"):
        verify_registration(thr, lock)


def test_register_then_verify_ok(paths):
    thr, lock = paths
    register_thresholds(thr, lock)
    info = verify_registration(thr, lock)
    assert info["sha256"] == thresholds_hash(thr)
    assert info["committed"] is False  # tmp files are not in the git repository


def test_editing_thresholds_after_registration_is_refused(paths):
    thr, lock = paths
    register_thresholds(thr, lock)
    _write(thr, {"acceptance_threshold": 0.5, "gate1": {"target_auroc_min": 0.75}})  # moved the goalposts
    with pytest.raises(PreregistrationError, match="changed since registration"):
        verify_registration(thr, lock)


def test_hash_ignores_formatting_and_key_order(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("x: 1\ny:\n  - 2\n  - 3\n", encoding="utf-8")
    b.write_text("# a comment\ny: [2, 3]\nx: 1\n", encoding="utf-8")
    assert thresholds_hash(a) == thresholds_hash(b)


def test_cannot_silently_reregister_different_thresholds(paths):
    thr, lock = paths
    register_thresholds(thr, lock)
    _write(thr, {"acceptance_threshold": 0.9, "gate1": {"target_auroc_min": 0.75}})
    with pytest.raises(PreregistrationError):
        register_thresholds(thr, lock)
    register_thresholds(thr, lock, force=True)  # explicit, deliberate override
    assert json.loads(lock.read_text())["sha256"] == thresholds_hash(thr)


def test_repo_thresholds_parse_and_engine_uses_them():
    from src.fuse.engine import ArbitrationEngine

    thr = load_thresholds()
    engine = ArbitrationEngine()
    assert engine.acceptance_threshold == thr.acceptance_threshold
    assert engine.abstention_threshold == thr.abstention_threshold
    assert canonical_hash(thr.model_dump()) == canonical_hash(load_thresholds().model_dump())
