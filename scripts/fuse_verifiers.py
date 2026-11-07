"""Stage 3-lite: does fusing Jev with an NLI verifier beat Jev alone? (zero API calls, calibration+validation only)

Out-of-fold, document-grouped 5-fold CV. Fusion = logistic regression on the two verifiers' logits (optionally plus a
per-dataset intercept). Reports the measured error correlation phi between the verifiers and the resulting effective
independent votes N_eff = k / (1 + (k-1) * phi), the quantity the repo's original design relied on.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate.metrics import auroc_score, brier_score, expected_calibration_error
from src.config import load_seeds, load_thresholds
from src.eval.bootstrap import cluster_bootstrap_multi, paired_cluster_bootstrap_test

CLIP, N_FOLDS = (0.01, 0.99), 5


def main() -> None:
    seeds, thr = load_seeds(), load_thresholds()
    run = sorted(glob.glob(str(ROOT / "results" / "runs" / "phase1-*")), key=os.path.getmtime)[-1]
    jev = pd.concat([pd.read_parquet(Path(run) / "records_cal.parquet"), pd.read_parquet(Path(run) / "records_val.parquet")])
    jev = jev[(jev["error"] == "") & jev["raw_p"].notna()][["id", "group", "subdataset", "label", "raw_p"]]
    df = jev.merge(pd.read_parquet(ROOT / "data" / "nli" / "scores.parquet")[["id", "p_entail"]], on="id").reset_index(drop=True)
    y, g, subs = df["label"].values.astype(int), df["group"].values, df["subdataset"].values
    lj, ln = logit(np.clip(df["raw_p"], *CLIP)), logit(np.clip(df["p_entail"], *CLIP))
    cols = sorted(df["subdataset"].unique())
    dom = pd.get_dummies(df["subdataset"]).reindex(columns=cols, fill_value=0).values.astype(float)

    uniq = np.array(sorted(set(g)))
    order = np.random.RandomState(seeds["split_seed"]).permutation(len(uniq))
    fo = {d: int(k % N_FOLDS) for k, d in zip(range(len(uniq)), uniq[order])}
    folds = np.array([fo[d] for d in g])

    def oof(X):
        p = np.zeros(len(y))
        for k in range(N_FOLDS):
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X[folds != k], y[folds != k])
            p[folds == k] = m.predict_proba(X[folds == k])[:, 1]
        return p

    arms = {
        "Jev only, domain-aware": oof(np.column_stack([lj, dom])),
        "NLI only, domain-aware": oof(np.column_stack([ln, dom])),
        "Jev + NLI fused, domain-agnostic": oof(np.column_stack([lj, ln])),
        "Jev + NLI fused, domain-aware": oof(np.column_stack([lj, ln, dom])),
    }
    nb = thr.phase1.n_bins
    fns = {"auroc": auroc_score, "brier": brier_score, "ece": lambda a, b: expected_calibration_error(a, b, nb)}
    print(f"{len(df)} claims, {df['group'].nunique()} documents, {N_FOLDS}-fold document-grouped CV (calibration+validation only)\n")
    print("| arm | pooled AUROC | macro within-dataset AUROC | Brier [95% CI] | ECE | paired dBrier vs Jev domain-aware (p) |\n|---|---|---|---|---|---|")
    base = arms["Jev only, domain-aware"]
    for name, p in arms.items():
        r = cluster_bootstrap_multi(g, y, p, fns, thr.phase1.bootstrap_iterations, thr.phase1.ci_level, seeds["bootstrap_seed"])
        macro = np.mean([auroc_score(y[subs == s], p[subs == s]) for s in cols if len(set(y[subs == s])) == 2])
        if name == "Jev only, domain-aware":
            cell = "-"
        else:
            d, pv = paired_cluster_bootstrap_test(g, y, base, p, lambda a, b: -brier_score(a, b), thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"])
            cell = f"{d:+.4f} ({pv:.3f})"
        print(f"| {name} | {r['auroc'][0]:.3f} | {macro:.3f} | {r['brier'][0]:.3f} [{r['brier'][1]:.3f}, {r['brier'][2]:.3f}] | {r['ece'][0]:.3f} | {cell} |")

    # fitted weights on all data (descriptive only), and effective independent votes from the measured error correlation
    m = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([lj, ln]), y)
    print(f"\nfitted fusion weights (all data, descriptive): Jev logit {m.coef_[0][0]:+.2f}, NLI logit {m.coef_[0][1]:+.2f}, intercept {m.intercept_[0]:+.2f}")
    ej, en = ((df["raw_p"] >= .5).astype(int) != y).astype(int), ((df["p_entail"] >= .5).astype(int) != y).astype(int)
    phi = float(np.corrcoef(ej, en)[0, 1])
    print(f"measured error correlation phi(Jev, NLI) = {phi:.3f}  ->  N_eff = 2 / (1 + phi) = {2 / (1 + phi):.2f} effective independent votes (of 2)")
    best_single = min(brier_score(y, arms["Jev only, domain-aware"]), brier_score(y, arms["NLI only, domain-aware"]))
    print(f"best single verifier Brier {best_single:.4f} vs fused-domain-aware {brier_score(y, arms['Jev + NLI fused, domain-aware']):.4f}")


if __name__ == "__main__":
    main()
