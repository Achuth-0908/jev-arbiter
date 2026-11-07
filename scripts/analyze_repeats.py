"""Offline analysis (zero API calls): does repeats=1 match repeats=3?

Replays the cached 3-repeat Jev responses for the calibration and validation splits and compares
the mean-of-3 score with the first repeat alone. Uses only calibration/validation data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibrate.metrics import auroc_score, brier_score, expected_calibration_error
from src.calibrate.scalers import IsotonicCalibrator
from src.clients.base import DecisionRequest
from src.clients.cache import CacheStore
from src.config import load_thresholds
from src.data.aggrefact import build_protocol_splits
from src.eval.bootstrap import cluster_bootstrap_multi, paired_cluster_bootstrap_test
from src.retrieval.bm25 import retrieve_evidence
from src.runs.spec import RunSpec
from src.verify.verifier import JevClaimVerifier


def replay(instances, spec, cache):
    """Return per-claim (group, label, p_mean3, p_first, p_mean2, cost_per_call) from cache only."""
    j, r = spec.jev, spec.retrieval
    rows, costs, missing = [], [], 0
    for inst in instances:
        snip, _ = retrieve_evidence(inst.evidence, inst.claim, r.words_per_chunk, r.overlap, r.top_k)
        req = DecisionRequest(model=j.model, decision_type="noul", state={"claim": inst.claim, "evidence": snip},
                              options=["Yes", "No"], criteria=JevClaimVerifier.DEFAULT_CRITERIA, repeats=j.repeats)
        hit = cache.get(req.cache_key(j.endpoint, mode="live"))
        if not hit:
            missing += 1
            continue
        data, _, cost = hit
        reps = [x["probabilities"]["Yes"] for x in data["results"]]
        lo, hi = j.clip_bounds
        c = lambda v: float(np.clip(round(v, 4), lo, hi))
        rows.append((str(inst.metadata["group"]), int(inst.label), c(np.mean(reps)), c(reps[0]), c(np.mean(reps[:2]))))
        costs.append(cost / max(1, len(reps)))
    return rows, costs, missing


def main() -> None:
    spec = RunSpec.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "runs" / "phase1.yaml")
    thr = load_thresholds()
    cal_i, val_i, _ = build_protocol_splits(spec.data.max_per_subdataset_dev, spec.data.max_per_subdataset_test)
    cache = CacheStore()
    cal, cc, mc = replay(cal_i, spec, cache)
    val, cv, mv = replay(val_i, spec, cache)
    print(f"replayed from cache: cal={len(cal)} (missing {mc}), val={len(val)} (missing {mv}); API calls made: 0")

    def arr(rows, k):
        return np.array([r[k] for r in rows])

    g, y = arr(val, 0), arr(val, 1).astype(int)
    nb = thr.phase1.n_bins
    fns = {"auroc": auroc_score, "brier": brier_score, "ece": lambda a, b: expected_calibration_error(a, b, nb)}
    print("\nVALIDATION, raw scores (95% cluster-bootstrap CI)")
    print(f"{'view':<14}{'AUROC':<26}{'Brier':<26}{'ECE':<26}")
    for name, k in (("mean of 3", 2), ("mean of 2", 4), ("repeat 1 only", 3)):
        res = cluster_bootstrap_multi(g, y, arr(val, k), fns, 500, seed=1337)
        print(f"{name:<14}" + "".join(f"{res[m][0]:.3f} [{res[m][1]:.3f},{res[m][2]:.3f}]".ljust(26) for m in ("auroc", "brier", "ece")))

    p3, p1 = arr(val, 2), arr(val, 3)
    print(f"\nmean |p(rep1) - p(mean3)| = {np.mean(np.abs(p1 - p3)):.4f}   decision flips at 0.5 = {np.mean((p1 >= .5) != (p3 >= .5)):.1%}")
    d, p = paired_cluster_bootstrap_test(g, y, p3, p1, auroc_score, 500, seed=1337)
    print(f"paired AUROC difference (rep1 - mean3) = {d:+.4f}   one-sided p(rep1 better) = {p:.2f}")

    # Full pipeline at repeats=1: refit isotonic on calibration repeat-1 scores, evaluate on validation.
    cy = arr(cal, 1).astype(int)
    print("\nISOTONIC refit on calibration, evaluated on validation")
    for name, k in (("mean of 3", 2), ("repeat 1 only", 3)):
        iso = IsotonicCalibrator(tuple(thr.clip_bounds)).fit(arr(cal, k), cy)
        pv = iso.transform(arr(val, k))
        print(f"{name:<14} ECE={expected_calibration_error(y, pv, nb):.4f}  Brier={brier_score(y, pv):.4f}  AUROC={auroc_score(y, pv):.4f}")

    per_call = float(np.mean(cc + cv))
    n_test = 2200
    print(f"\nmeasured cost per live call: ${per_call:.6f}")
    print(f"test split ({n_test} claims): repeats=3 ~ ${n_test * 3 * per_call:.3f}   repeats=1 ~ ${n_test * per_call:.3f}")


if __name__ == "__main__":
    main()
