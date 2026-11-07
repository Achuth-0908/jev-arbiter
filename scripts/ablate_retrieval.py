"""Stage 1b step 2: is Jev's weakness (e.g. ExpertQA) a retrieval failure? Live Jev calls, hard-capped.

Re-scores calibration/validation claims with different evidence retrieval and compares RAW Jev scores to
the baseline retrieval (top_k=2, 120-word chunks) already in the latest Phase 1 run. The test split is never
touched. No calibration is involved, so the comparison isolates the effect of the evidence.

usage: python scripts/ablate_retrieval.py --subset expertqa|val --variants top5,chunk250 --budget-usd 0.03
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate.metrics import auroc_score, brier_score
from src.clients.cache import CacheStore
from src.clients.jev import FatalAPIError, JevDecisionsClient
from src.config import load_seeds, load_thresholds
from src.data.aggrefact import build_protocol_splits
from src.eval.bootstrap import cluster_bootstrap_multi, paired_cluster_bootstrap_test
from src.retrieval.bm25 import retrieve_evidence
from src.runs.phase1 import score_split, valid
from src.runs.spec import RetrievalSpec, RunSpec
from src.verify.verifier import JevClaimVerifier

BASE_RETRIEVAL = RetrievalSpec(words_per_chunk=120, overlap=20, top_k=2)
LENIENT = ("Is the claim supported by the evidence text? Count it as supported if the evidence states it, "
           "paraphrases it, or clearly implies it; answer false only if the evidence is silent on it or contradicts it.")
# name -> (retrieval, criteria override or None)
VARIANTS = {
    "top5": (RetrievalSpec(words_per_chunk=120, overlap=20, top_k=5), None),       # more evidence passages
    "chunk250": (RetrievalSpec(words_per_chunk=250, overlap=40, top_k=2), None),   # larger passages
    "lenient": (BASE_RETRIEVAL, LENIENT),                                          # same evidence, looser wording
    "fulldoc": (RetrievalSpec(words_per_chunk=1_000_000, overlap=20, top_k=1), None),  # the whole document, no retrieval
}


def baseline_frame() -> pd.DataFrame:
    run = sorted(glob.glob(str(ROOT / "results" / "runs" / "phase1-*")), key=os.path.getmtime)[-1]
    df = pd.concat([pd.read_parquet(Path(run) / "records_cal.parquet"),
                    pd.read_parquet(Path(run) / "records_val.parquet")], ignore_index=True)
    return valid(df).set_index("id")


async def run_variant(name, subset_insts, budget, log):
    retrieval, criteria = VARIANTS[name]
    spec = RunSpec(name=f"abl-{name}", mode="live", retrieval=retrieval)
    client = JevDecisionsClient(cache_store=CacheStore(), concurrency_limit=spec.jev.concurrency,
                                max_spend_usd=budget)
    if client.mock_mode:
        raise RuntimeError("live mode required (no API key?)")
    try:
        verifier = JevClaimVerifier(client=client, repeats=1)
        if criteria:
            verifier.DEFAULT_CRITERIA = criteria  # instance override; verify() uses it when no criteria is passed
        df = await score_split(subset_insts, f"abl-{name}", verifier, spec, log)
    finally:
        await client.aclose()
    return df, client


def report(name, subset, base, var, n_boot, seed, only_ids=None, label=""):
    ids = sorted(set(base.index) & set(var["id"]) & (set(only_ids) if only_ids is not None else set(base.index)))
    if len(ids) < 20 or base.loc[ids, "label"].nunique() < 2:
        print(f"\n## {name} on {subset}{label}: n={len(ids)} too small for a comparison")
        return
    v = var.set_index("id").loc[ids]
    b = base.loc[ids]
    y, g = b["label"].values.astype(int), b["group"].values
    pb, pv = b["raw_p"].values, v["raw_p"].values
    fns = {"auroc": auroc_score, "brier": brier_score}
    rb = cluster_bootstrap_multi(g, y, pb, fns, n_boot, 0.95, seed)
    rv = cluster_bootstrap_multi(g, y, pv, fns, n_boot, 0.95, seed)
    d, p = paired_cluster_bootstrap_test(g, y, pb, pv, auroc_score, n_boot, seed)
    f = lambda r, m: f"{r[m][0]:.3f} [{r[m][1]:.3f}, {r[m][2]:.3f}]"
    print(f"\n## {name} on {subset} (n={len(ids)})")
    print(f"- baseline (top_k=2, 120w): AUROC {f(rb, 'auroc')}, Brier {f(rb, 'brier')}")
    print(f"- {name:<24}: AUROC {f(rv, 'auroc')}, Brier {f(rv, 'brier')}")
    print(f"- paired AUROC difference (variant - baseline) = {d:+.3f}, one-sided p(variant better) = {p:.3f}")
    print(f"- mean |p_variant - p_baseline| = {np.mean(np.abs(pv - pb)):.3f}")
    if subset == "val":
        sub = b["subdataset"].values
        print("- per-dataset AUROC (baseline -> variant):")
        gains = []
        for s in sorted(set(sub)):
            m = sub == s
            if len(set(y[m])) == 2:
                a0, a1 = auroc_score(y[m], pb[m]), auroc_score(y[m], pv[m])
                gains.append(a1 - a0)
                print(f"    {s:<18} {a0:.3f} -> {a1:.3f}  ({a1 - a0:+.3f})")
        print(f"  macro within-dataset AUROC change: {np.mean(gains):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["expertqa", "val"], default="expertqa")
    ap.add_argument("--variants", default="top5,chunk250")
    ap.add_argument("--budget-usd", type=float, default=0.03, help="hard cap PER variant")
    a = ap.parse_args()
    thr, seeds = load_seeds(), None
    thr, seeds = load_thresholds(), load_seeds()
    cal, val, _ = build_protocol_splits(200, 200)
    insts = [i for i in (cal + val) if i.subdataset == "ExpertQA"] if a.subset == "expertqa" else list(val)
    base = baseline_frame()
    for name in a.variants.split(","):
        print(f"\n[{name}] scoring {len(insts)} claims live (cap ${a.budget_usd})", file=sys.stderr, flush=True)
        try:
            df, client = asyncio.run(run_variant(name, insts, a.budget_usd, lambda m: print(m, file=sys.stderr, flush=True)))
        except FatalAPIError as e:
            print(f"ABORTED ({name}): {e}  (finished claims stay cached; re-run resumes free)")
            continue
        out = ROOT / "results" / "analysis" / "ablation"
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / f"{name}_{a.subset}.parquet", index=False)
        ok = valid(df)
        spent = float(ok.loc[~ok["cached"], "cost_usd"].sum())
        print(f"[{name}] ok={len(ok)} failed={int((df['error'] != '').sum())} spent=${spent:.4f} "
              f"| est. full-val cost ${spent / max(1, len(ok)) * 738:.3f}")
        report(name, a.subset, base, df, thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"])
        retrieval, criteria = VARIANTS[name]
        if criteria is None:  # evidence-changing variant: also isolate claims whose evidence really differed
            changed = [i.id for i in insts if retrieve_evidence(i.evidence, i.claim, 120, 20, 2)[0]
                       != retrieve_evidence(i.evidence, i.claim, retrieval.words_per_chunk, retrieval.overlap, retrieval.top_k)[0]]
            report(name, a.subset, base, df, thr.phase1.bootstrap_iterations, seeds["bootstrap_seed"],
                   only_ids=changed, label=" [only claims whose evidence changed]")


if __name__ == "__main__":
    main()
