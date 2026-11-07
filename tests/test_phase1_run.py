"""End-to-end Phase 1 pipeline in mock mode (offline): artifacts, idempotency, test-split guard."""

import json

import pandas as pd
import pytest

from src.clients.cache import CacheStore
from src.config import PreregistrationError, load_thresholds
from src.data.aggrefact import create_grouped_splits, doc_group_id
from src.data.loaders import FactClaimInstance
from src.runs.phase1 import run_phase1
from src.runs.spec import RunSpec


def _instances(n_docs: int = 40, claims_per_doc: int = 4):
    out = []
    docs = []
    for d in range(n_docs):
        words = [f"d{d}w{k}" for k in range(150)]
        docs.append(words)
    for d, words in enumerate(docs):
        doc = " ".join(words)
        gid = doc_group_id(doc)
        sub = ["A", "B", "C"][d % 3]
        for c in range(claims_per_doc):
            if c % 2 == 0:  # supported: lifted straight from the document
                claim, label = " ".join(words[20 + c * 7 : 28 + c * 7]), 1
            else:           # unsupported: lifted from a different document
                other = docs[(d + 1 + c) % n_docs]
                claim, label = " ".join(other[10:18]), 0
            out.append(FactClaimInstance(id=f"{d}-{c}", claim=claim, evidence=doc, label=label,
                                         subdataset=sub, metadata={"group": gid}))
    return out


@pytest.fixture
def setup(tmp_path):
    thr = load_thresholds()
    thr = thr.model_copy(update={"phase1": thr.phase1.model_copy(update={
        "bootstrap_iterations": 60, "nondeterminism_probe_n": 20, "wording_probe_n": 20})})
    splits = create_grouped_splits(_instances(), ratios=(0.4, 0.3, 0.3), seed=42)
    spec = RunSpec(name="t", mode="mock", output_dir=str(tmp_path / "runs"))
    cache = CacheStore(db_path=str(tmp_path / "cache.db"))
    return thr, splits, spec, cache


@pytest.mark.asyncio
async def test_run_writes_artifacts_and_never_touches_test_by_default(setup):
    thr, splits, spec, cache = setup
    res = await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    from pathlib import Path

    out = Path(res["output_dir"])
    for f in ("manifest.json", "metrics.json", "report.md", "records_cal.parquet", "records_val.parquet"):
        assert (out / f).exists(), f
    assert not (out / "records_test.parquet").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["test_scored"] is False and man["mode"] == "mock"
    assert man["split_sizes"]["test"] > 0           # the split exists...
    assert "test" not in res and "gate1_test" not in res  # ...but was never evaluated
    val = res["validation"]["selected"]
    assert val["auroc"]["lo"] <= val["auroc"]["point"] <= val["auroc"]["hi"]
    assert res["selected_calibrator"] in ("temperature", "isotonic")
    rec = pd.read_parquet(out / "records_val.parquet")
    assert {"raw_p", "p_selected", "group", "label", "cached", "error"} <= set(rec.columns)


@pytest.mark.asyncio
async def test_rerun_is_idempotent_with_zero_new_work(setup):
    thr, splits, spec, cache = setup
    first = await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    misses_after_first = cache.misses
    second = await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    assert cache.misses == misses_after_first            # no cache miss => no new external call
    assert first["manifest"]["run_id"] == second["manifest"]["run_id"]
    assert first["validation"] == second["validation"]
    assert first["selected_calibrator"] == second["selected_calibrator"]
    assert second["manifest"]["cost"]["spent_this_run_usd"] == 0.0


@pytest.mark.asyncio
async def test_include_test_is_refused_without_valid_registration(setup, monkeypatch):
    thr, splits, spec, cache = setup
    spec.include_test = True

    def refuse():
        raise PreregistrationError("Thresholds are not registered.")

    monkeypatch.setattr("src.runs.phase1.verify_registration", refuse)
    with pytest.raises(PreregistrationError):
        await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    assert cache.misses == 0  # refused before a single claim was scored


@pytest.mark.asyncio
async def test_include_test_with_registration_scores_test_and_reports_gate(setup, monkeypatch):
    thr, splits, spec, cache = setup
    spec.include_test = True
    monkeypatch.setattr("src.runs.phase1.verify_registration",
                        lambda: {"sha256": "x", "registered_utc": "t", "committed": False})
    res = await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    assert res["manifest"]["test_scored"] is True
    assert set(res["gate1_test"]["criteria"]) >= {"auroc", "ece", "brier", "nondeterminism_divergence"}
    assert (res["output_dir"] and __import__("pathlib").Path(res["output_dir"], "records_test.parquet").exists())


@pytest.mark.asyncio
async def test_live_mode_refuses_to_silently_fall_back_to_mock(setup, monkeypatch):
    thr, splits, spec, cache = setup
    spec.mode = "live"
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("MOCK_API_MODE", "true")
    with pytest.raises(RuntimeError, match="fell back to mock"):
        await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)


@pytest.mark.asyncio
async def test_billing_error_aborts_the_run_instead_of_producing_partial_metrics(setup, monkeypatch):
    from src.clients.jev import FatalAPIError
    from src.verify.verifier import JevClaimVerifier

    thr, splits, spec, cache = setup

    async def boom(self, *a, **k):
        raise FatalAPIError("Decisions API returned non-retryable HTTP 402 (insufficient credits)")

    monkeypatch.setattr(JevClaimVerifier, "verify", boom)
    with pytest.raises(FatalAPIError):
        await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)


@pytest.mark.asyncio
async def test_run_refuses_a_protocol_that_differs_from_the_registered_repeats(setup):
    thr, splits, spec, cache = setup
    spec.jev.repeats = thr.phase1.repeats + 2
    with pytest.raises(ValueError, match="differs from the registered protocol"):
        await run_phase1(spec, thresholds=thr, splits=splits, cache=cache, log=lambda m: None)
    assert cache.misses == 0
