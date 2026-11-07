"""Stage 1b, step 1: does base-rate / feature-aware calibration beat the global calibrator? (zero API calls)

Uses ONLY calibration + validation claims (the test split is never loaded for scoring) and the cached
Jev probabilities from the latest Phase 1 run. Every arm is scored out-of-fold with document-grouped
K-fold CV, so no source document is ever in both the fit and the evaluation fold.

Pre-specified before looking at results: PRIMARY metric = Brier (the failing Gate 1 criterion);
SECONDARY = macro-averaged per-dataset ECE. Comparisons are paired against global isotonic.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate.metrics import auroc_score, brier_score, expected_calibration_error
from src.calibrate.scalers import IsotonicCalibrator, TemperatureScaler
from src.config import load_seeds, load_thresholds
from src.data.aggrefact import build_protocol_splits
from src.eval.bootstrap import cluster_bootstrap_multi, paired_cluster_bootstrap_test
from src.retrieval.bm25 import retrieve_evidence

N_FOLDS = 5
CLIP = (0.01, 0.99)


def load_frame() -> pd.DataFrame:
    run = sorted(glob.glob(str(ROOT / "results" / "runs" / "phase1-*")), key=os.path.getmtime)[-1]
    df = pd.concat(
        [pd.read_parquet(Path(run) / "records_cal.parquet"), pd.read_parquet(Path(run) / "records_val.parquet")],
        ignore_index=True,
    )
    df = df[(df["error"] == "") & df["raw_p"].notna()].reset_index(drop=True)
    cal_i, val_i, _ = build_protocol_splits(200, 200)  # test list is built but never scored or used here
    inst = {i.id: i for i in cal_i + val_i}
    feats = []
    for rid in df["id"]:
        it = inst[rid]
        _, bm = retrieve_evidence(it.evidence, it.claim, 120, 20, 2)
        feats.append((len(it.claim.split()), len(it.evidence.split()), bm))
    df[["claim_words", "doc_words", "bm25"]] = np.array(feats)
    return df


def fold_ids(groups: np.ndarray, seed: int) -> np.ndarray:
    """Assign whole documents to folds (seeded), so no document spans train and eval."""
    uniq = np.array(sorted(set(groups)))
    order = np.random.RandomState(seed).permutation(len(uniq))
    fold_of = {g: int(k % N_FOLDS) for k, g in zip(range(len(uniq)), uniq[order])}
    return np.array([fold_of[g] for g in groups])


def x_features(d: pd.DataFrame) -> np.ndarray:
    p = np.clip(d["raw_p"].values, *CLIP)
    return np.column_stack([logit(p), np.log1p(d["claim_words"]), np.log1p(d["doc_words"]), np.minimum(d["bm25"], 30) / 30])


# ---- arms: (train_df, test_df) -> probabilities for test_df ---------------------------------------
def arm_raw(tr, te):
    return np.clip(te["raw_p"].values, *CLIP)


def arm_iso_global(tr, te):
    return IsotonicCalibrator(CLIP).fit(tr["raw_p"].values, tr["label"].values).transform(te["raw_p"].values)


def arm_temp_global(tr, te):
    return TemperatureScaler(CLIP).fit(tr["raw_p"].values, tr["label"].values).transform(te["raw_p"].values)


def arm_lr_features(tr, te):  # domain-AGNOSTIC: only things observable per claim at inference
    sc = StandardScaler().fit(x_features(tr))
    m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(x_features(tr)), tr["label"].values)
    return m.predict_proba(sc.transform(x_features(te)))[:, 1]


def _dom(d, cols):
    return pd.get_dummies(d["subdataset"]).reindex(columns=cols, fill_value=0).values.astype(float)


def arm_lr_domain(tr, te):  # domain-AWARE: logit(p) plus a per-dataset intercept (prior shift)
    cols = sorted(tr["subdataset"].unique())
    f = lambda d: np.column_stack([logit(np.clip(d["raw_p"].values, *CLIP)), _dom(d, cols)])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(f(tr), tr["label"].values)
    return m.predict_proba(f(te))[:, 1]


def arm_iso_perdataset(tr, te):  # domain-AWARE: separate monotone map per dataset (global fallback)
    out = np.zeros(len(te))
    glob_iso = IsotonicCalibrator(CLIP).fit(tr["raw_p"].values, tr["label"].values)
    for sub in te["subdataset"].unique():
        a, b = tr[tr["subdataset"] == sub], (te["subdataset"] == sub).values
        if len(a) >= 30 and a["label"].nunique() == 2:
            out[b] = IsotonicCalibrator(CLIP).fit(a["raw_p"].values, a["label"].values).transform(te.loc[b, "raw_p"].values)
        else:
            out[b] = glob_iso.transform(te.loc[b, "raw_p"].values)
    return out


def arm_prior_only(tr, te):  # CONTROL: ignores Jev entirely; predicts each dataset's training positive rate
    rates = tr.groupby("subdataset")["label"].mean()
    return np.clip(te["subdataset"].map(rates).fillna(tr["label"].mean()).values, *CLIP)


ARMS = [
    ("CONTROL: dataset base rate only (ignores Jev)", "control", arm_prior_only),
    ("raw (uncalibrated)", "-", arm_raw),
    ("global isotonic  [current]", "agnostic", arm_iso_global),
    ("global temperature", "agnostic", arm_temp_global),
    ("logistic on logit(p)+claim/doc length+BM25", "agnostic", arm_lr_features),
    ("logistic on logit(p)+dataset intercept", "domain-aware", arm_lr_domain),
    ("per-dataset isotonic", "domain-aware", arm_iso_perdataset),
]


def main() -> None:
    seeds, thr = load_seeds(), load_thresholds()
    nb = thr.phase1.n_bins
    df = load_frame()
    folds = fold_ids(df["group"].values, seeds["split_seed"])
    y, g, subs = df["label"].values.astype(int), df["group"].values, df["subdataset"].values
    print(f"{len(df)} claims (calibration+validation only), {df['group'].nunique()} documents, {N_FOLDS}-fold document-grouped CV\n")

    oof = {}
    for name, _, fn in ARMS:
        p = np.zeros(len(df))
        for k in range(N_FOLDS):
            tr, te = df[folds != k], df[folds == k]
            p[(folds == k)] = fn(tr, te)
        oof[name] = p

    fns = {"auroc": auroc_score, "brier": brier_score, "ece": lambda a, b: expected_calibration_error(a, b, nb)}
    raw_ece = {s: expected_calibration_error(y[subs == s], oof["raw (uncalibrated)"][subs == s], nb) for s in np.unique(subs)}
    base = oof["global isotonic  [current]"]

    lines = ["# Stage 1b step 1: calibration arms (out-of-fold, calibration+validation only)\n",
             f"{len(df)} claims, {df['group'].nunique()} documents, {N_FOLDS}-fold document-grouped CV, "
             f"{thr.phase1.bootstrap_iterations}-iteration cluster bootstrap. PRIMARY: Brier. SECONDARY: macro per-dataset ECE.\n",
             "| arm | type | Brier [95% CI] | ECE [95% CI] | AUROC | macro-ECE | datasets with ECE < raw | paired dBrier vs global isotonic (p) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, kind, _ in ARMS:
        p = oof[name]
        r = cluster_bootstrap_multi(g, y, p, fns, thr.phase1.bootstrap_iterations, thr.phase1.ci_level, seeds["bootstrap_seed"])
        per = {s: expected_calibration_error(y[subs == s], p[subs == s], nb) for s in np.unique(subs)}
        better = sum(per[s] < raw_ece[s] for s in per)
        if name.startswith("global isotonic"):
            dcell = "-"
        else:
            d, pv = paired_cluster_bootstrap_test(g, y, base, p, lambda a, b: -brier_score(a, b), thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"])
            dcell = f"{d:+.4f} ({pv:.3f})"
        f = lambda m: f"{r[m][0]:.3f} [{r[m][1]:.3f}, {r[m][2]:.3f}]"
        lines.append(f"| {name} | {kind} | {f('brier')} | {f('ece')} | {r['auroc'][0]:.3f} | {np.mean(list(per.values())):.3f} | {better}/11 | {dcell} |")
    lines.append("\ndBrier = (Brier of global isotonic) - (Brier of the arm): positive means the arm is better; p is one-sided, UNADJUSTED for the 5 comparisons against global isotonic (Bonferroni: multiply p by 5).")
    out = "\n".join(lines) + "\n"
    print(out)
    d = ROOT / "results" / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "calibration_1b.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
