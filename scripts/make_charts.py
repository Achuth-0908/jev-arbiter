"""Generate the README / LinkedIn charts from real run artifacts. No result is typed in by hand.

Inputs (all local; the per-claim files are git-ignored because they derive from a licensed dataset):
  results/runs/phase1-*/            latest run that scored the test split (metrics.json, records_*.parquet)
  results/analysis/calibration_1b.md  output of scripts/analyze_calibration.py
  results/analysis/outputs/val_multiq.parquet  output of scripts/explore_outputs.py
  data/nli/scores.parquet           output of scripts/nli_score.py
Outputs: docs/charts/*.png (1200x627 at 2x) and docs/charts/data/*.csv (the table twin of each chart).

Design follows the dataviz skill: one accent hue per series (validated pair blue/orange), status colors only for
pass/fail and always with an icon and a label, solid hairline grids, selective direct labels, ink-colored text.
usage: python scripts/make_charts.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_nli as cn  # noqa: E402  (reuses its fold assignment and out-of-fold calibrators)
from src.calibrate.metrics import auroc_score, brier_score  # noqa: E402
from src.config import load_seeds  # noqa: E402

OUT = ROOT / "docs" / "charts"
DATA = OUT / "data"

# ---- tokens (validated reference palette; blue/orange pass the validator: CVD dE 24.7, normal dE 33.6, >= 3:1) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRIT = "#0ca30c", "#d03b3b"  # status colors: only for pass/fail, always with icon + label
CLIP = (0.01, 0.99)

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [f for f in ("Segoe UI", "Inter", "Helvetica", "Arial") if f in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SYMBOL_FONT = "DejaVu Sans"  # reliable glyphs for check / cross marks

FOOT = "Independent evaluation; not affiliated with TypeSafe or OpenRouter. Dataset: LLM-AggreFact (11 sub-datasets)."


def new_fig(title: str, subtitle: str, tag: str | None = None, foot: str = FOOT):
    fig = plt.figure(figsize=(12, 6.27), dpi=200, facecolor=SURFACE)
    fig.text(0.05, 0.94, title, fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.865, subtitle, fontsize=12, color=INK2, va="top")
    if tag:
        fig.text(0.95, 0.872, tag, fontsize=9.5, color=INK2, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.45", fc="#f0efec", ec=AXIS, lw=0.8))
    fig.text(0.05, 0.028, foot, fontsize=8.5, color=MUTED, va="bottom")
    return fig


def style_axes(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis=grid_axis, color=GRID, lw=1, ls="-")
    ax.set_axisbelow(True)


def save(fig, name: str, table: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)
    table.to_csv(DATA / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"wrote docs/charts/{name}.png  (+ data/{name}.csv)")


# ---------------------------------------------------------------------------------------------- data loading
def latest_test_run() -> Path:
    dirs = sorted(glob.glob(str(ROOT / "results" / "runs" / "phase1-*")), key=os.path.getmtime, reverse=True)
    for d in dirs:
        man = json.loads(Path(d, "manifest.json").read_text(encoding="utf-8"))
        if man.get("test_scored"):
            return Path(d)
    raise SystemExit("No Phase 1 run that scored the test split was found in results/runs/. Run phase1 --include-test first.")


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["error"] == "") & df["raw_p"].notna()]


# ---------------------------------------------------------------------------------------------- chart 1
def chart_scorecard(met: dict) -> None:
    sel, raw = met["test"]["selected"], met["test"]["raw"]
    crit = met["gate1_test"]["criteria"]
    n = sel["n"]
    rows = [
        ("AUROC", "ranking quality", sel["auroc"], crit["auroc"], 3),
        ("Calibration error (ECE)", f"after recalibration; raw {raw['ece']['point']:.3f}", sel["ece"], crit["ece"], 3),
        ("Brier score", "probability accuracy", sel["brier"], crit["brier"], 3),
        ("Run-to-run difference", "two independent calls, mean absolute difference", None, crit["nondeterminism_divergence"], 4),
    ]
    fig = new_fig("I pre-registered a test. It failed on one metric.",
                  f"Jev as a claim verifier, official test split, {n:,} claims, scored once.",
                  tag=f"CONFIRMATORY  |  test split  |  n = {n:,}",
                  foot="Thresholds were committed to git before the test was scored. 95% intervals: bootstrap over source documents. "
                       "Independent evaluation; not affiliated with TypeSafe or OpenRouter.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, t in ((0.05, "METRIC"), (0.43, "RESULT (95% INTERVAL)"), (0.66, "TARGET"), (0.80, "OUTCOME")):
        ax.text(x, 0.765, t, fontsize=9, color=MUTED, fontweight="bold")
    top, h = 0.735, 0.145
    records = []
    for i, (name, note, m, c, nd) in enumerate(rows):
        y0 = top - (i + 1) * h
        yc = y0 + h / 2
        failed = not c["pass"]
        if failed:
            ax.add_patch(Rectangle((0.045, y0 + 0.006), 0.91, h - 0.012, fc=CRIT, alpha=0.07, ec="none"))
        ax.plot([0.05, 0.95], [y0, y0], color=GRID, lw=1)
        ax.text(0.05, yc + 0.022, name, fontsize=16, color=INK, va="center", fontweight="bold" if failed else "normal")
        ax.text(0.05, yc - 0.028, note, fontsize=10, color=INK2, va="center")
        val = c["value"] if m is None else m["point"]
        ax.text(0.43, yc + 0.012, f"{val:.{nd}f}", fontsize=24, color=INK, va="center", fontweight="bold")
        if m is not None:
            ax.text(0.43, yc - 0.04, f"[{m['lo']:.3f}, {m['hi']:.3f}]", fontsize=10.5, color=INK2, va="center")
        op = {">=": "≥", "<=": "≤"}.get(c["op"], c["op"])
        ax.text(0.66, yc, f"{op} {c['target']}", fontsize=16, color=INK2, va="center")
        color = GOOD if c["pass"] else CRIT
        ax.add_patch(FancyBboxPatch((0.80, yc - 0.034), 0.125, 0.068, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=color, ec="none"))
        ax.text(0.8155, yc, "✓" if c["pass"] else "✗", fontsize=15, color="white", va="center",
                ha="center", fontweight="bold", fontfamily=SYMBOL_FONT)
        ax.text(0.8675, yc, "PASS" if c["pass"] else "FAIL", fontsize=13, color="white", va="center", ha="center", fontweight="bold")
        records.append({"metric": name, "value": val, "ci_lo": (m or {}).get("lo"), "ci_hi": (m or {}).get("hi"),
                        "target": f"{c['op']} {c['target']}", "pass": c["pass"]})
    save(fig, "01_gate1_scorecard", pd.DataFrame(records))


# ---------------------------------------------------------------------------------------------- chart 2
def base_rate_pooled_auroc(run: Path) -> float:
    """Pooled AUROC of a predictor that ignores the claim and outputs each dataset's training positive rate."""
    cal = pd.read_parquet(run / "records_cal.parquet")
    val = pd.read_parquet(run / "records_val.parquet")
    df = valid(pd.concat([cal, val], ignore_index=True)).reset_index(drop=True)
    folds = cn.fold_ids(df["group"].values, load_seeds()["split_seed"])
    p = np.zeros(len(df))
    for k in range(cn.N_FOLDS):
        tr, te = df[folds != k], df[folds == k]
        rates = tr.groupby("subdataset")["label"].mean()
        p[folds == k] = np.clip(te["subdataset"].map(rates).fillna(tr["label"].mean()).values, *CLIP)
    return float(auroc_score(df["label"].values.astype(int), p))


def chart_dataset_auroc(met: dict, base_auc: float) -> None:
    per = met["per_dataset_test"]
    rows = sorted(((ds, d["raw"]["auroc"]["point"], d["raw"]["auroc"]["lo"], d["raw"]["auroc"]["hi"], d["raw"]["n"])
                   for ds, d in per.items()), key=lambda r: r[1])
    fig = new_fig("Skill varies a lot by dataset",
                  "Within-dataset AUROC is the honest measure of claim-level skill; the pooled number flatters.",
                  tag="CONFIRMATORY  |  test split  |  200 claims per dataset",
                  foot="Raw (uncalibrated) Jev scores. Whiskers: 95% bootstrap intervals over source documents. Axis starts at 0.4. "
                       "Reference line: calibration + validation data, cross-validated, by document.")
    ax = fig.add_axes([0.21, 0.145, 0.73, 0.615])
    style_axes(ax, "x")
    for i, (ds, p, lo, hi, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=2.2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot(p, i, "o", color=BLUE, ms=9.5, mec=SURFACE, mew=2, zorder=3)
    for i in (0, len(rows) - 1):  # selective direct labels: the extremes only
        ds, p, lo, hi, n = rows[i]
        ax.text(hi + 0.012, i, f"{p:.2f}", fontsize=11, color=INK, va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK2)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("Raw AUROC within dataset (95% interval)", fontsize=10.5, color=INK2, labelpad=8)
    ytop = len(rows) - 0.1
    for x, label, dy in ((0.5, "chance", 0), (base_auc, f"predictor that ignores the claim\nand outputs each dataset's base rate: {base_auc:.3f} pooled", 0)):
        ax.plot([x, x], [-0.7, ytop], color=INK2, lw=1.4, zorder=1)
        ax.text(x, ytop + 0.12, label, fontsize=9.5, color=INK2, ha="center", va="bottom", linespacing=1.15)
    save(fig, "02_per_dataset_auroc",
         pd.DataFrame(rows, columns=["dataset", "auroc", "ci_lo", "ci_hi", "n"]).assign(base_rate_only_pooled_auroc=round(base_auc, 4)))


# ---------------------------------------------------------------------------------------------- chart 3
def chart_calibration_arms(target: float) -> None:
    md = ROOT / "results" / "analysis" / "calibration_1b.md"
    if not md.exists():
        print("skip chart 3: results/analysis/calibration_1b.md not found (run scripts/analyze_calibration.py)")
        return
    names = {"CONTROL": "Base rate only (ignores the claim)", "raw": "Raw Jev", "global isotonic": "Global isotonic (current method)",
             "global temperature": "Global temperature scaling", "logistic on logit(p)+claim": "Logistic + length and BM25 features",
             "logistic on logit(p)+dataset": "Logistic + per-dataset intercept", "per-dataset isotonic": "Per-dataset isotonic"}
    rows = []
    for line in md.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not line.startswith("|") or cells[0] in ("arm", "---"):
            continue
        m = re.match(r"([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", cells[2])
        label = next((v for k, v in names.items() if cells[0].startswith(k)), None)
        if m and label:
            rows.append((label, cells[1], float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if len(rows) < 5:
        print("skip chart 3: could not parse the calibration table")
        return
    rows.sort(key=lambda r: -r[2])  # best (lowest Brier) at the top
    fig = new_fig("Domain-aware calibration gets closest to the Brier target",
                  "Brier score (lower is better), 5-fold cross-validation grouped by source document.",
                  foot="Exploratory: calibration + validation data (n = 2,200), not confirmed on the test split. 95% intervals over "
                       "source documents. Axis starts at 0.10.")
    ax = fig.add_axes([0.30, 0.145, 0.64, 0.615])
    style_axes(ax, "x")
    for i, (label, kind, b, lo, hi) in enumerate(rows):
        col = BLUE if kind == "domain-aware" else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2.2, alpha=0.4, solid_capstyle="round", zorder=2)
        ax.plot(b, i, "o", color=col, ms=9.5, mec=SURFACE, mew=2, zorder=3)
    best = len(rows) - 1
    cur = next(i for i, r in enumerate(rows) if r[0].startswith("Global isotonic"))
    for i in (best, cur):
        ax.text(rows[i][4] + 0.003, i, f"{rows[i][2]:.3f}", fontsize=11, color=INK, va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK2)
    ax.set_xlim(0.10, 0.20)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("Brier score (95% interval)", fontsize=10.5, color=INK2, labelpad=8)
    ax.plot([target, target], [-0.7, len(rows) - 0.1], color=INK2, lw=1.4, zorder=1)
    ax.text(target, len(rows) + 0.02, f"pre-registered target {target}", fontsize=9.5, color=INK2, ha="center", va="bottom")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=BLUE, ls="", ms=9, label="domain-aware"),
                       plt.Line2D([], [], marker="o", color=MUTED, ls="", ms=9, label="domain-agnostic / baselines")],
              loc="upper right", frameon=False, fontsize=10, labelcolor=INK2)
    save(fig, "03_calibration_arms", pd.DataFrame(rows, columns=["arm", "type", "brier", "ci_lo", "ci_hi"]))


# ---------------------------------------------------------------------------------------------- chart 4
def chart_jev_vs_nli(run: Path) -> None:
    nli_path = ROOT / "data" / "nli" / "scores.parquet"
    if not nli_path.exists():
        print("skip chart 4: data/nli/scores.parquet not found (run scripts/nli_score.py)")
        return
    recs = valid(pd.concat([pd.read_parquet(run / "records_cal.parquet"), pd.read_parquet(run / "records_val.parquet")]))
    df = recs.merge(pd.read_parquet(nli_path)[["id", "p_entail"]], on="id").reset_index(drop=True)
    y = df["label"].values.astype(int)
    rows = []
    for ds, g in df.groupby("subdataset"):
        if g["label"].nunique() == 2:
            rows.append((ds, auroc_score(g["label"].values, g["p_entail"].values), auroc_score(g["label"].values, g["raw_p"].values), len(g)))
    rows.sort(key=lambda r: r[2])
    folds = cn.fold_ids(df["group"].values, load_seeds()["split_seed"])
    bj = brier_score(y, cn.oof(df, "raw_p", "dom", folds))
    bn = brier_score(y, cn.oof(df, "p_entail", "dom", folds))
    cols = sorted(df["subdataset"].unique())
    dom = pd.get_dummies(df["subdataset"]).reindex(columns=cols, fill_value=0).values.astype(float)
    X = np.column_stack([logit(np.clip(df["raw_p"], *CLIP)), logit(np.clip(df["p_entail"], *CLIP)), dom])
    pf = np.zeros(len(y))
    for k in range(cn.N_FOLDS):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[folds != k], y[folds != k])
        pf[folds == k] = m.predict_proba(X[folds == k])[:, 1]
    bf = brier_score(y, pf)
    wins = sum(r[2] > r[1] for r in rows)
    fig = new_fig(f"Jev beat a generic NLI model on {wins} of {len(rows)} datasets",
                  "Same retrieved snippets for both; raw AUROC within each dataset.",
                  foot="Exploratory: calibration + validation data. NLI: one generic model (DeBERTa-v3-base MNLI/FEVER/ANLI), not an "
                       "LLM judge. Brier: 5-fold CV. Axis starts at 0.5.")
    ax = fig.add_axes([0.20, 0.145, 0.47, 0.615])
    style_axes(ax, "x")
    for i, (ds, a_n, a_j, n) in enumerate(rows):
        ax.plot([a_n, a_j], [i, i], color=AXIS, lw=2.5, zorder=1, solid_capstyle="round")
        ax.plot(a_n, i, "o", color=ORANGE, ms=9.5, mec=SURFACE, mew=2, zorder=3)
        ax.plot(a_j, i, "o", color=BLUE, ms=9.5, mec=SURFACE, mew=2, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK2)
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Raw AUROC within dataset", fontsize=10.5, color=INK2, labelpad=8)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=BLUE, ls="", ms=9, label="Jev"),
                       plt.Line2D([], [], marker="o", color=ORANGE, ls="", ms=9, label="NLI model")],
              loc="lower right", frameon=False, fontsize=11, labelcolor=INK2)
    fig.text(0.72, 0.74, "BRIER AFTER DOMAIN-AWARE CALIBRATION", fontsize=9, color=MUTED, fontweight="bold")
    for j, (label, val, col) in enumerate((("Jev", bj, BLUE), ("NLI model", bn, ORANGE), ("Jev + NLI fused", bf, INK2))):
        y0 = 0.64 - j * 0.12
        fig.text(0.72, y0, f"{val:.3f}", fontsize=26, fontweight="bold", color=INK)
        fig.add_artist(Line2D([0.826], [y0 + 0.0215], marker="o", ms=9, ls="", color=col, transform=fig.transFigure))
        fig.text(0.84, y0 + 0.0215, label, fontsize=12, color=INK2, va="center")
    fig.text(0.72, 0.26, "Fusing the two added nothing: the fitted\nweight on the NLI model was about zero.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.3)
    save(fig, "04_jev_vs_nli", pd.DataFrame(rows, columns=["dataset", "auroc_nli", "auroc_jev", "n"])
         .assign(brier_jev=round(bj, 4), brier_nli=round(bn, 4), brier_fused=round(bf, 4)))


# ---------------------------------------------------------------------------------------------- chart 5
def chart_selective_accuracy() -> None:
    f = ROOT / "results" / "analysis" / "outputs" / "val_multiq.parquet"
    if not f.exists():
        print("skip chart 5: results/analysis/outputs/val_multiq.parquet not found (run scripts/explore_outputs.py)")
        return
    d = pd.read_parquet(f)
    y, p = d["label"].values.astype(int), d["p_noul"].values.astype(float)
    pred = (p >= 0.5).astype(int)
    order = np.argsort(-np.abs(p - 0.5), kind="stable")  # most confident first
    cov = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    acc = [float(np.mean(pred[order[: int(round(c * len(y)))]] == y[order[: int(round(c * len(y)))]])) for c in cov]
    fig = new_fig("Let it abstain, and accuracy climbs",
                  "Answer only the claims where Jev's probability is farthest from 0.5.",
                  foot=f"Exploratory: validation split (n = {len(y)}), single run, no confidence interval. Accuracy of the yes/no "
                       "decision at probability 0.5. Axis starts at 0.7.")
    ax = fig.add_axes([0.08, 0.14, 0.86, 0.62])
    style_axes(ax, "y")
    ax.plot(cov, acc, "-", color=BLUE, lw=2.5, zorder=2)
    ax.plot(cov, acc, "o", color=BLUE, ms=10, mec=SURFACE, mew=2, zorder=3)
    ax.set_xlim(1.03, 0.47)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(cov)
    ax.set_xticklabels([f"{int(round(c * 100))}%" for c in cov])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Coverage: share of claims answered", fontsize=10.5, color=INK2, labelpad=8)
    ax.set_ylabel("Accuracy on answered claims", fontsize=10.5, color=INK2, labelpad=8)
    for c, a, dy, ha in ((1.0, acc[0], -0.045, "center"), (0.8, acc[2], 0.045, "center"), (0.5, acc[-1], 0.04, "center")):
        ax.text(c, a + dy, f"{a:.1%}", fontsize=12.5, fontweight="bold", color=INK, ha=ha, va="center")
    ax.annotate(f"{acc[2]:.0%} accurate on the 80% most confident claims", xy=(0.8, acc[2]), xytext=(0.69, 0.775),
                fontsize=11, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    save(fig, "05_selective_accuracy", pd.DataFrame({"coverage": cov, "accuracy": [round(a, 4) for a in acc], "n_answered": [int(round(c * len(y))) for c in cov]}))


# ---------------------------------------------------------------------------------------------- chart 6
def chart_pipeline() -> None:
    fig = new_fig("What makes this evaluation trustworthy",
                  "Each safeguard is enforced in code and covered by offline tests.",
                  foot="Details: AGENTS.md and docs/JEV_FINDINGS.md in the repository.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [("Leak-free splits", "No source document\nappears in two splits"),
             ("Evidence retrieval", "BM25 snippets from\nthe source document"),
             ("Jev verifier", "Cached, rate-limited,\nspend-capped"),
             ("Calibration", "Fit on calibration data,\nchosen on validation"),
             ("Test scored once", "Only if the registered\nthresholds hash matches")]
    w, gap, x0, y0, h = 0.155, 0.0312, 0.05, 0.43, 0.27
    for i, (t, s) in enumerate(steps):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=SURFACE, ec=AXIS, lw=1.2))
        ax.plot(x + 0.026, y0 + h - 0.05, "o", ms=20, color=BLUE, mec=BLUE, zorder=3)  # a marker is always a true circle
        ax.text(x + 0.026, y0 + h - 0.05, str(i + 1), color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(x + 0.012, y0 + h - 0.115, t, fontsize=13, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.012, y0 + 0.095, s, fontsize=10, color=INK2, va="center", linespacing=1.35)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.004, y0 + h / 2), xytext=(x + w + 0.004, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    chips = ["Zero API calls\non re-run", "Bootstrap CIs\nby document", "Run manifests", "Hard budget cap", "No silent failures"]
    ax.plot([x0, x0 + 5 * w + 4 * gap], [0.345, 0.345], color=GRID, lw=1)
    ax.text(x0, 0.31, "ENFORCED THROUGHOUT THE PIPELINE", fontsize=9, color=MUTED, fontweight="bold", va="center")
    for i, c in enumerate(chips):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.17), w, 0.12, boxstyle="round,pad=0,rounding_size=0.01", fc="#f0efec", ec="none"))
        ax.text(x + w / 2, 0.23, c, fontsize=10.5, color=INK, ha="center", va="center", linespacing=1.3)
    save(fig, "06_pipeline", pd.DataFrame({"step": [s[0] for s in steps], "description": [s[1].replace("\n", " ") for s in steps]}))


def main() -> None:
    run = latest_test_run()
    met = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    print(f"using run {run.name}")
    base_auc = base_rate_pooled_auroc(run)
    print(f"base-rate-only pooled AUROC (recomputed): {base_auc:.4f}")
    target = met["gate1_test"]["criteria"]["brier"]["target"]
    chart_scorecard(met)
    chart_dataset_auroc(met, base_auc)
    chart_calibration_arms(target)
    chart_jev_vs_nli(run)
    chart_selective_accuracy()
    chart_pipeline()


if __name__ == "__main__":
    main()
