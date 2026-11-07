"""Anchor Jev against an NLI baseline on calibration + validation claims (zero API calls, test never used).

Both verifiers see the SAME retrieved snippet. Compared raw and after identical out-of-fold, document-grouped
calibration (global isotonic, and domain-aware logistic with a per-dataset intercept).
Pre-specified primary comparison: macro within-dataset AUROC (claim-level discrimination, not inflated by base rates),
then Brier after domain-aware calibration.
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
from src.calibrate.scalers import IsotonicCalibrator
from src.config import load_seeds, load_thresholds
from src.eval.bootstrap import cluster_bootstrap_multi, paired_cluster_bootstrap_test

CLIP = (0.01, 0.99)
N_FOLDS = 5


def fold_ids(groups, seed):
    uniq = np.array(sorted(set(groups)))
    order = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = {g: int(k % N_FOLDS) for k, g in zip(range(len(uniq)), uniq[order])}
    return np.array([fold_of[g] for g in groups])


def macro_auroc(df, col):
    v = [auroc_score(g["label"].values, g[col].values) for _, g in df.groupby("subdataset") if g["label"].nunique() == 2]
    return float(np.mean(v))


def oof(df, col, kind, folds):
    out = np.zeros(len(df))
    cols = sorted(df["subdataset"].unique())
    dom = lambda d: pd.get_dummies(d["subdataset"]).reindex(columns=cols, fill_value=0).values.astype(float)
    for k in range(N_FOLDS):
        tr, te = df[folds != k], df[folds == k]
        if kind == "raw":
            out[folds == k] = np.clip(te[col].values, *CLIP)
        elif kind == "iso":
            out[folds == k] = IsotonicCalibrator(CLIP).fit(tr[col].values, tr["label"].values).transform(te[col].values)
        else:  # domain-aware logistic
            f = lambda d: np.column_stack([logit(np.clip(d[col].values, *CLIP)), dom(d)])
            out[folds == k] = LogisticRegression(C=1.0, max_iter=1000).fit(f(tr), tr["label"].values).predict_proba(f(te))[:, 1]
    return out


def main() -> None:
    seeds, thr = load_seeds(), load_thresholds()
    run = sorted(glob.glob(str(ROOT / "results" / "runs" / "phase1-*")), key=os.path.getmtime)[-1]
    jev = pd.concat([pd.read_parquet(Path(run) / "records_cal.parquet"), pd.read_parquet(Path(run) / "records_val.parquet")])
    jev = jev[(jev["error"] == "") & jev["raw_p"].notna()][["id", "group", "subdataset", "label", "raw_p"]]
    nli = pd.read_parquet(ROOT / "data" / "nli" / "scores.parquet")[["id", "p_entail", "p_neutral", "p_contradiction"]]
    df = jev.merge(nli, on="id", how="inner").reset_index(drop=True)
    print(f"{len(df)} claims with both Jev and NLI scores (calibration+validation only), {df['group'].nunique()} documents\n")
    folds = fold_ids(df["group"].values, seeds["split_seed"])
    y, g, nb = df["label"].values.astype(int), df["group"].values, thr.phase1.n_bins
    fns = {"auroc": auroc_score, "brier": brier_score, "ece": lambda a, b: expected_calibration_error(a, b, nb)}

    arms = {}
    for who, col in (("Jev", "raw_p"), ("NLI", "p_entail")):
        for kind, label in (("raw", "raw"), ("iso", "global isotonic"), ("dom", "domain-aware")):
            arms[f"{who} {label}"] = oof(df, col, kind, folds)

    lines = ["| verifier / calibration | pooled AUROC | macro within-dataset AUROC | Brier [95% CI] | ECE |", "|---|---|---|---|---|"]
    for name, p in arms.items():
        r = cluster_bootstrap_multi(g, y, p, fns, thr.phase1.bootstrap_iterations, thr.phase1.ci_level, seeds["bootstrap_seed"])
        tmp = df.assign(_p=p)
        lines.append(f"| {name} | {r['auroc'][0]:.3f} | {macro_auroc(tmp, '_p'):.3f} | "
                     f"{r['brier'][0]:.3f} [{r['brier'][1]:.3f}, {r['brier'][2]:.3f}] | {r['ece'][0]:.3f} |")
    print("\n".join(lines))

    print("\nper-dataset AUROC (raw scores)")
    print("| dataset | n | Jev | NLI | Jev - NLI |\n|---|---|---|---|---|")
    for s, gr in df.groupby("subdataset"):
        if gr["label"].nunique() == 2:
            a, b = auroc_score(gr["label"], gr["raw_p"]), auroc_score(gr["label"], gr["p_entail"])
            print(f"| {s} | {len(gr)} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")

    d, p = paired_cluster_bootstrap_test(g, y, arms["NLI domain-aware"], arms["Jev domain-aware"],
                                         lambda a, b: -brier_score(a, b), thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"])
    print(f"\npaired: domain-aware Brier, NLI minus Jev = {d:+.4f} (positive = Jev better), one-sided p = {p:.3f}")
    d2, p2 = paired_cluster_bootstrap_test(g, y, df["p_entail"].values, df["raw_p"].values, auroc_score,
                                           thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"])
    print(f"paired: raw pooled AUROC, Jev minus NLI = {d2:+.4f}, one-sided p(Jev better) = {p2:.3f}")
    corr = np.corrcoef(logit(np.clip(df["raw_p"], *CLIP)), logit(np.clip(df["p_entail"], *CLIP)))[0, 1]
    print(f"corr(logit Jev, logit NLI) = {corr:.3f}")
    # do they make different mistakes? (useful for the multi-agent / fusion idea later)
    ej, en = ((df["raw_p"] >= .5).astype(int) != y).astype(int), ((df["p_entail"] >= .5).astype(int) != y).astype(int)
    print(f"error rates: Jev {ej.mean():.3f}, NLI {en.mean():.3f}; error correlation (phi) = {np.corrcoef(ej, en)[0, 1]:.3f}; "
          f"both wrong {np.mean(ej & en):.3f}, only Jev wrong {np.mean(ej & (1 - en)):.3f}, only NLI wrong {np.mean((1 - ej) & en):.3f}")


if __name__ == "__main__":
    main()
