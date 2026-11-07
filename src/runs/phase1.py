"""Phase 1 pipeline: Jev calibration probe on LLM-AggreFact.

calibration split -> fit scalers          (never used for evaluation)
validation split  -> select calibrator, report metrics with cluster-bootstrap CIs
test split        -> scored ONLY when include_test is set AND the pre-registration verifies
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..calibrate.metrics import auroc_score, brier_score, expected_calibration_error
from ..calibrate.scalers import IsotonicCalibrator, TemperatureScaler
from ..clients.cache import CacheStore
from ..clients.jev import FatalAPIError, JevDecisionsClient
from ..config import (
    REPO_ROOT,
    Thresholds,
    canonical_hash,
    git_info,
    load_seeds,
    load_thresholds,
    verify_registration,
)
from ..data.aggrefact import assert_no_group_overlap, build_protocol_splits, doc_group_id
from ..data.loaders import FactClaimInstance
from ..eval.bootstrap import cluster_bootstrap_multi
from ..retrieval.bm25 import retrieve_evidence
from ..verify.verifier import JevClaimVerifier
from .spec import RunSpec

Splits = Tuple[List[FactClaimInstance], List[FactClaimInstance], List[FactClaimInstance]]
Log = Callable[[str], None]


def _stderr_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fingerprint(instances: Sequence[FactClaimInstance]) -> str:
    return hashlib.sha256("\n".join(sorted(i.id for i in instances)).encode("utf-8")).hexdigest()[:16]


def _group(inst: FactClaimInstance) -> str:
    return str(inst.metadata.get("group") or doc_group_id(inst.evidence))


# --------------------------------------------------------------------------- scoring

async def score_split(
    instances: Sequence[FactClaimInstance],
    split: str,
    verifier: JevClaimVerifier,
    spec: RunSpec,
    log: Log = _stderr_log,
) -> pd.DataFrame:
    """Score every claim concurrently (bounded). Failures are recorded, never imputed."""
    sem = asyncio.Semaphore(spec.jev.concurrency)
    total = len(instances)
    rows: List[Optional[Dict[str, Any]]] = [None] * total
    step = max(1, total // 10)
    state = {"done": 0}
    t0 = time.time()
    r = spec.retrieval

    async def one(i: int, inst: FactClaimInstance) -> None:
        base = {"id": inst.id, "split": split, "subdataset": inst.subdataset,
                "group": _group(inst), "label": int(inst.label)}
        async with sem:
            try:
                snippet, _ = retrieve_evidence(inst.evidence, inst.claim, r.words_per_chunk, r.overlap, r.top_k)
                res = await verifier.verify(inst.claim, snippet)
                rows[i] = {**base, "raw_p": res.raw_p_supported, "variance": res.variance_across_repeats,
                           "cost_usd": res.cost_usd, "latency_ms": res.latency_ms,
                           "cached": res.cached, "resolved_version": res.resolved_version, "error": ""}
            except FatalAPIError:
                raise  # account/auth problem: abort the whole run, do not record per-claim failures
            except Exception as e:  # noqa: BLE001 - record the class only, never request details
                rows[i] = {**base, "raw_p": float("nan"), "variance": float("nan"), "cost_usd": 0.0,
                           "latency_ms": 0.0, "cached": False, "resolved_version": "", "error": type(e).__name__}
        state["done"] += 1
        if state["done"] % step == 0 or state["done"] == total:
            log(f"  [{split}] {state['done']}/{total} claims  ({time.time() - t0:.0f}s)")

    await asyncio.gather(*(one(i, x) for i, x in enumerate(instances)))
    return pd.DataFrame(rows)


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["error"] == "") & df["raw_p"].notna()]


# --------------------------------------------------------------------------- metrics

def metric_fns(n_bins: int) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "auroc": auroc_score,
        "brier": brier_score,
        "ece": lambda y, p: expected_calibration_error(y, p, n_bins=n_bins),
        "accuracy": lambda y, p: float(np.mean((p >= 0.5).astype(int) == y)),
    }


def metrics_with_ci(df: pd.DataFrame, col: str, thr: Thresholds, seed: int) -> Dict[str, Any]:
    ok = valid(df)
    if len(ok) == 0 or ok["label"].nunique() < 2:
        return {"n": int(len(ok)), "n_clusters": int(ok["group"].nunique())}
    res = cluster_bootstrap_multi(
        ok["group"].values, ok["label"].values.astype(int), ok[col].values.astype(float),
        metric_fns(thr.phase1.n_bins), thr.phase1.bootstrap_iterations, thr.phase1.ci_level, seed,
    )
    out: Dict[str, Any] = {"n": int(len(ok)), "n_clusters": int(ok["group"].nunique()),
                           "positive_rate": float(ok["label"].mean())}
    out.update({k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in res.items()})
    return out


def per_dataset_table(df: pd.DataFrame, cols: Dict[str, str], thr: Thresholds, seed: int) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for sub in sorted(df["subdataset"].unique()):
        part = df[df["subdataset"] == sub]
        table[sub] = {name: metrics_with_ci(part, col, thr, seed) for name, col in cols.items()}
    return table


# --------------------------------------------------------------------------- calibration

def fit_and_select(
    cal: pd.DataFrame, val: pd.DataFrame, thr: Thresholds
) -> Tuple[Dict[str, Any], Dict[str, float], str]:
    """Fit calibrators on `cal`; choose among them by validation ECE (pre-registered rule)."""
    c, v = valid(cal), valid(val)
    if len(c) < 20 or c["label"].nunique() < 2:
        raise RuntimeError(f"Too few valid calibration claims to fit calibrators (n={len(c)}).")
    makers = {"temperature": TemperatureScaler, "isotonic": IsotonicCalibrator}
    scalers: Dict[str, Any] = {}
    for name in thr.phase1.calibrators:
        scalers[name] = makers[name](tuple(thr.clip_bounds)).fit(c["raw_p"].values, c["label"].values)
    val_ece = {
        name: expected_calibration_error(v["label"].values, sc.transform(v["raw_p"].values), thr.phase1.n_bins)
        for name, sc in scalers.items()
    }
    selected = min(thr.phase1.calibrators, key=lambda n: (val_ece[n], thr.phase1.calibrators.index(n)))
    return scalers, val_ece, selected


def apply_calibrators(df: pd.DataFrame, scalers: Dict[str, Any], selected: str) -> pd.DataFrame:
    out = df.copy()
    ok = out["raw_p"].notna()
    for name, sc in scalers.items():
        out[f"p_{name}"] = np.nan
        out.loc[ok, f"p_{name}"] = sc.transform(out.loc[ok, "raw_p"].values)
    out["p_selected"] = out[f"p_{selected}"]
    return out


# --------------------------------------------------------------------------- probes

async def _probe(
    instances: Sequence[FactClaimInstance], verifier: JevClaimVerifier, spec: RunSpec, kind: str
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(spec.jev.concurrency)
    r = spec.retrieval
    pairs: List[Tuple[float, float]] = []
    spent = [0.0]
    errors = [0]

    async def one(inst: FactClaimInstance) -> None:
        async with sem:
            try:
                snippet, _ = retrieve_evidence(inst.evidence, inst.claim, r.words_per_chunk, r.overlap, r.top_k)
                base = await verifier.verify(inst.claim, snippet)
                if kind == "nondeterminism":
                    other = await verifier.verify(inst.claim, snippet, cache_salt="nd1")
                else:
                    other = await verifier.verify(inst.claim, snippet, criteria=verifier.ALT_CRITERIA)
                pairs.append((base.raw_p_supported, other.raw_p_supported))
                spent[0] += 0.0 if other.cached else other.cost_usd
            except FatalAPIError:
                raise
            except Exception:  # noqa: BLE001
                errors[0] += 1

    await asyncio.gather(*(one(i) for i in instances))
    if not pairs:
        return {"n": 0, "errors": errors[0]}
    a, b = np.array(pairs).T
    d = np.abs(a - b)
    return {
        "n": len(pairs), "errors": errors[0],
        "mean_abs_diff": float(d.mean()), "median_abs_diff": float(np.median(d)),
        "p95_abs_diff": float(np.percentile(d, 95)), "max_abs_diff": float(d.max()),
        "decision_flip_rate": float(np.mean((a >= 0.5) != (b >= 0.5))),
        "spent_usd": spent[0],
    }


# --------------------------------------------------------------------------- gate

def gate1_verdict(point: Dict[str, float], nondeterminism: Optional[float], thr: Thresholds) -> Dict[str, Any]:
    g = thr.gate1
    checks = {
        "auroc": (point["auroc"], g.target_auroc_min, ">=", point["auroc"] >= g.target_auroc_min),
        "ece": (point["ece"], g.target_ece_max, "<=", point["ece"] <= g.target_ece_max),
        "brier": (point["brier"], g.target_brier_max, "<=", point["brier"] <= g.target_brier_max),
    }
    if nondeterminism is not None:
        checks["nondeterminism_divergence"] = (
            nondeterminism, g.nondeterminism_divergence_max, "<=", nondeterminism <= g.nondeterminism_divergence_max
        )
    rows = {k: {"value": v[0], "target": v[1], "op": v[2], "pass": bool(v[3])} for k, v in checks.items()}
    return {"criteria": rows, "all_pass": all(r["pass"] for r in rows.values())}


# --------------------------------------------------------------------------- reporting

def _clean(o: Any) -> Any:
    """JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _fmt(m: Dict[str, Any], key: str) -> str:
    v = m.get(key)
    if not isinstance(v, dict) or v.get("point") is None:
        return "n/a"
    lo, hi = v.get("lo"), v.get("hi")
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{v['point']:.3f}"
    return f"{v['point']:.3f} [{lo:.3f}, {hi:.3f}]"


def render_report(res: Dict[str, Any]) -> str:
    man, thr = res["manifest"], res["thresholds"]
    L: List[str] = []
    L.append(f"# Phase 1 run `{man['run_id']}`  ({man['mode']} mode)\n")
    L.append(f"- Jev model requested: `{man['jev_model']}`; resolved by API: `{', '.join(man['resolved_versions']) or 'n/a'}`")
    L.append(f"- Splits: cal={man['split_sizes']['cal']}, val={man['split_sizes']['val']}, test={man['split_sizes']['test']}"
             f" (test scored: **{man['test_scored']}**)")
    reg = man["registration"]
    if reg:
        flag = "" if reg.get("committed") else "  **(NOT committed to git: not independently timestamped)**"
        L.append(f"- Registration: lock verified (sha256 {reg['sha256'][:12]}..., registered {reg['registered_utc']}); "
                 f"committed to git: {reg.get('committed')}{flag}")
    else:
        L.append("- Registration: not required (test not scored)")
    L.append(f"- Jev repeats per claim: {man['spec']['jev']['repeats']}; code version {str(man['git']['commit'])[:10]} "
             f"(working tree dirty: {man['git']['dirty']})")
    L.append(f"- Bootstrap: {thr['phase1']['bootstrap_iterations']} iterations, clustered by source document, "
             f"{int(thr['phase1']['ci_level'] * 100)}% CI")
    L.append(f"- Effective independent votes N_eff: 1.0 (single verifier; not applicable to Phase 1)\n")

    sel = res["selected_calibrator"]
    L.append("## Calibrator selection (validation only)")
    L.append("| calibrator | validation ECE |\n|---|---|")
    for k, v in res["validation_ece_by_calibrator"].items():
        L.append(f"| {k}{' **(selected)**' if k == sel else ''} | {v:.4f} |")
    L.append(f"\nFitted temperature T = {res['fitted_temperature']:.3f}\n")

    def block(title: str, section: Dict[str, Any]) -> None:
        L.append(f"## {title}")
        L.append("| scores | n | ECE | Brier | AUROC | accuracy |\n|---|---|---|---|---|---|")
        for name in ("raw", "temperature", "isotonic", "selected"):
            m = section.get(name)
            if m:
                L.append(f"| {name} | {m['n']} | {_fmt(m, 'ece')} | {_fmt(m, 'brier')} | {_fmt(m, 'auroc')} | {_fmt(m, 'accuracy')} |")
        L.append("")

    block("Validation (informational; not the gate)", res["validation"])
    if res.get("test"):
        block("TEST (Gate 1 evaluation)", res["test"])

    for title, key in (("Per-dataset, validation (raw vs selected)", "per_dataset_validation"),
                       ("Per-dataset, TEST (raw vs selected)", "per_dataset_test")):
        tbl = res.get(key)
        if not tbl:
            continue
        L.append(f"## {title}")
        L.append("| dataset | n | pos rate | raw ECE | sel ECE | raw AUROC | sel Brier |\n|---|---|---|---|---|---|---|")
        for sub, d in tbl.items():
            raw, sl = d["raw"], d["selected"]
            pr = raw.get("positive_rate")
            L.append(f"| {sub} | {raw['n']} | {pr:.2f} | {_fmt(raw, 'ece')} | {_fmt(sl, 'ece')} | "
                     f"{_fmt(raw, 'auroc')} | {_fmt(sl, 'brier')} |" if pr is not None else f"| {sub} | {raw['n']} | n/a | | | | |")
        L.append("")

    p = res["probes"]
    L.append("## Robustness probes (calibration-split sample)")
    for name, d in (("Nondeterminism (fresh independent call vs original)", p["nondeterminism"]),
                    ("Wording invariance (paraphrased instructions)", p["wording"])):
        if d.get("n"):
            warn = f"  **WARNING: {d['errors']} probe calls failed; n is partial**" if d.get("errors") else ""
            L.append(f"- **{name}**: n={d['n']}, mean |dp|={d['mean_abs_diff']:.4f}, p95={d['p95_abs_diff']:.3f}, "
                     f"max={d['max_abs_diff']:.3f}, decision flips={d['decision_flip_rate']:.1%}{warn}")
        else:
            L.append(f"- **{name}**: no data (errors={d.get('errors', 0)})")
    L.append("")

    for title, key in (("Gate 1 preview on VALIDATION (not binding)", "gate1_validation_preview"),
                       ("Gate 1 verdict on TEST (binding)", "gate1_test")):
        g = res.get(key)
        if not g:
            continue
        L.append(f"## {title}")
        L.append("| criterion | value | target | pass |\n|---|---|---|---|")
        for k, r in g["criteria"].items():
            L.append(f"| {k} | {r['value']:.4f} | {r['op']} {r['target']} | {'PASS' if r['pass'] else 'FAIL'} |")
        L.append(f"\n**Overall: {'PASS' if g['all_pass'] else 'FAIL'}**\n")

    c = man["cost"]
    L.append("## Operations")
    L.append(f"- Claims scored: {c['claims_scored']} ({c['claims_failed']} failed); network calls this run: {c['network_calls']}; answered from cached larger entries: {c['derived_from_cache']}")
    L.append(f"- Spent this run: ${c['spent_this_run_usd']:.4f}; cold-run equivalent: ${c['cold_run_equivalent_usd']:.4f}")
    L.append(f"- Wall time: {man['timings_s']['total']:.1f}s")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- orchestration

async def run_phase1(
    spec: RunSpec,
    thresholds: Optional[Thresholds] = None,
    splits: Optional[Splits] = None,
    cache: Optional[CacheStore] = None,
    log: Log = _stderr_log,
) -> Dict[str, Any]:
    t_start = time.time()
    thr = thresholds or load_thresholds()
    seeds = load_seeds()
    if spec.jev.repeats != thr.phase1.repeats:
        raise ValueError(f"spec jev.repeats={spec.jev.repeats} differs from the registered protocol "
                         f"phase1.repeats={thr.phase1.repeats}; refusing to run a different protocol")

    # Fail before spending anything if the test split is requested without a valid registration.
    registration: Optional[Dict[str, Any]] = verify_registration() if spec.include_test else None

    if splits is None:
        splits = build_protocol_splits(
            spec.data.max_per_subdataset_dev, spec.data.max_per_subdataset_test, subsets=spec.data.subsets
        )
    cal_i, val_i, test_i = splits
    assert_no_group_overlap(cal_i, val_i, test_i)

    fps = {"cal": fingerprint(cal_i), "val": fingerprint(val_i), "test": fingerprint(test_i)}
    run_id = canonical_hash({"spec": spec.spec_hash(), "thr": thr.model_dump(), "seeds": seeds, "fps": fps})[:12]
    log(f"run {spec.name}-{run_id}: mode={spec.mode} cal={len(cal_i)} val={len(val_i)} test={len(test_i)}")

    cache = cache or CacheStore()
    j = spec.jev
    client = JevDecisionsClient(
        endpoint=j.endpoint, model=j.model, cache_store=cache, concurrency_limit=j.concurrency,
        timeout_seconds=j.timeout_seconds, max_retries=j.max_retries, backoff_factor=j.backoff_factor,
        mock_mode=True if spec.mode == "mock" else None, requests_per_second=j.requests_per_second,
        max_spend_usd=spec.budget_usd, derive_from_repeats=j.derive_from_repeats,
    )
    if spec.mode == "live" and client.mock_mode:
        raise RuntimeError("Run requests live mode but the client fell back to mock "
                           "(OPENROUTER_API_KEY missing or MOCK_API_MODE=true). Refusing to continue.")
    verifier = JevClaimVerifier(client=client, repeats=j.repeats, clip_bounds=j.clip_bounds)

    timings: Dict[str, float] = {}
    t = time.time()
    cal = await score_split(cal_i, "cal", verifier, spec, log)
    val = await score_split(val_i, "val", verifier, spec, log)
    timings["score_cal_val"] = time.time() - t

    fail_rate = float((cal["error"] != "").mean()) if len(cal) else 0.0
    fail_rate_v = float((val["error"] != "").mean()) if len(val) else 0.0
    if max(fail_rate, fail_rate_v) > 0.01:
        log(f"WARNING: failure rate above 1% (cal {fail_rate:.1%}, val {fail_rate_v:.1%}); see records for error classes")

    scalers, val_ece, selected = fit_and_select(cal, val, thr)
    cal = apply_calibrators(cal, scalers, selected)
    val = apply_calibrators(val, scalers, selected)
    boot_seed = seeds["bootstrap_seed"]

    t = time.time()
    cols = {"raw": "raw_p", **{n: f"p_{n}" for n in scalers}, "selected": "p_selected"}
    validation = {name: metrics_with_ci(val, col, thr, boot_seed) for name, col in cols.items()}
    per_ds_val = per_dataset_table(val, {"raw": "raw_p", "selected": "p_selected"}, thr, boot_seed)
    timings["bootstrap_val"] = time.time() - t

    # Probes on a seeded sample of CALIBRATION claims only.
    rng = np.random.RandomState(seeds["sampling_seed"])
    pool = sorted(cal_i, key=lambda i: i.id)
    n_probe = min(max(thr.phase1.nondeterminism_probe_n, thr.phase1.wording_probe_n), len(pool))
    sample = [pool[k] for k in sorted(rng.choice(len(pool), size=n_probe, replace=False))]
    t = time.time()
    nd = await _probe(sample[: thr.phase1.nondeterminism_probe_n], verifier, spec, "nondeterminism")
    wd = await _probe(sample[: thr.phase1.wording_probe_n], verifier, spec, "wording")
    timings["probes"] = time.time() - t
    nd_value = nd.get("mean_abs_diff")

    result: Dict[str, Any] = {
        "selected_calibrator": selected,
        "validation_ece_by_calibrator": val_ece,
        "fitted_temperature": float(scalers["temperature"].temperature) if "temperature" in scalers else float("nan"),
        "validation": validation,
        "per_dataset_validation": per_ds_val,
        "probes": {"nondeterminism": nd, "wording": wd},
        "gate1_validation_preview": gate1_verdict(
            {k: validation["selected"][k]["point"] for k in ("auroc", "ece", "brier")}, nd_value, thr
        ) if validation["selected"].get("auroc") else None,
        "thresholds": thr.model_dump(),
    }

    frames = {"cal": cal, "val": val}
    test_scored = False
    if spec.include_test:
        t = time.time()
        test = await score_split(test_i, "test", verifier, spec, log)
        test = apply_calibrators(test, scalers, selected)
        frames["test"] = test
        result["test"] = {name: metrics_with_ci(test, col, thr, boot_seed) for name, col in cols.items()}
        result["per_dataset_test"] = per_dataset_table(test, {"raw": "raw_p", "selected": "p_selected"}, thr, boot_seed)
        result["gate1_test"] = gate1_verdict(
            {k: result["test"]["selected"][k]["point"] for k in ("auroc", "ece", "brier")}, nd_value, thr
        )
        test_scored = True
        timings["score_and_eval_test"] = time.time() - t

    all_rows = pd.concat(frames.values(), ignore_index=True)
    ok = valid(all_rows)
    resolved = sorted(set(ok["resolved_version"]) - {""})
    timings["total"] = time.time() - t_start
    manifest = {
        "run_id": run_id, "name": spec.name, "mode": spec.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": spec.model_dump(), "spec_hash": spec.spec_hash(), "seeds": seeds,
        "jev_model": j.model, "resolved_versions": resolved,
        "split_sizes": {"cal": len(cal_i), "val": len(val_i), "test": len(test_i)},
        "split_fingerprints": fps, "test_scored": test_scored, "registration": registration,
        "git": git_info(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "timings_s": timings,
        "cost": {
            "claims_scored": int(len(ok)), "claims_failed": int((all_rows["error"] != "").sum()),
            "network_calls": int(client.network_calls), "derived_from_cache": int(client.derived_from_cache),
            "budget_usd": spec.budget_usd, "client_spent_usd": float(client.spent_usd),
            "spent_this_run_usd": float(ok.loc[~ok["cached"], "cost_usd"].sum() + nd.get("spent_usd", 0.0) + wd.get("spent_usd", 0.0)),
            "cold_run_equivalent_usd": float(ok["cost_usd"].sum()),
        },
        "probe_errors": {"nondeterminism": nd.get("errors", 0), "wording": wd.get("errors", 0)},
        "error_classes": {k: int(v) for k, v in all_rows.loc[all_rows["error"] != "", "error"].value_counts().items()},
    }
    result["manifest"] = manifest

    out_root = Path(spec.output_dir)
    out_dir = (out_root if out_root.is_absolute() else REPO_ROOT / out_root) / f"{spec.name}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(out_dir / f"records_{name}.parquet", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(_clean(manifest), indent=2), encoding="utf-8")
    metrics_doc = {k: v for k, v in result.items() if k not in ("manifest", "thresholds")}
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics_doc), indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(_clean(result)), encoding="utf-8")
    result["output_dir"] = str(out_dir)
    await client.aclose()
    log(f"done in {timings['total']:.1f}s -> {out_dir}")
    return result
