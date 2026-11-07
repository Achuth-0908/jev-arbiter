"""Explore Jev's other output types on VALIDATION claims: noul + choice + score in ONE call.

Questions per call (same state as the baseline: claim + BM25 snippets):
  is_supported   noul    - identical to the Phase 1 baseline question
  support_level  choice  - supported / unsupported / contradicted (per-option probabilities + confidence)
  support_score  score   - 4 ordered levels (probability-weighted score + distribution + confidence)

Goes through the content-addressed cache (key = endpoint, mode, model, state, questions). Hard spend cap.
Test split is never touched. usage: python scripts/explore_outputs.py --budget-usd 0.04
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate.metrics import auroc_score, brier_score
from src.clients.base import canonical_json_hash
from src.clients.cache import CacheStore
from src.clients.jev import FatalAPIError, JevDecisionsClient
from src.data.aggrefact import build_protocol_splits
from src.eval.bootstrap import cluster_bootstrap_multi
from src.retrieval.bm25 import retrieve_evidence
from src.verify.verifier import JevClaimVerifier

CRIT = JevClaimVerifier.DEFAULT_CRITERIA


def questions_for(claim: str) -> dict:
    return {
        "is_supported": {
            "type": "noul", "instructions": CRIT,
            "criteria": {
                "true": f"The claim '{claim}' is directly and strictly supported by the evidence.",
                "false": f"The claim '{claim}' is not supported or is contradicted by the evidence.",
            },
        },
        "support_level": {
            "type": "choice",
            "instructions": "Decide how the evidence text relates to the claim.",
            "criteria": {
                "supported": "The evidence states, paraphrases or clearly implies the whole claim.",
                "unsupported": "The evidence is silent on, or insufficient for, at least part of the claim.",
                "contradicted": "The evidence contradicts the claim.",
            },
        },
        "support_score": {
            "type": "score",
            "instructions": "Rate how fully the evidence text supports the claim.",
            "criteria": [
                "Contradicted by the evidence",
                "Not supported: the evidence is silent or insufficient",
                "Partly supported: some parts of the claim are supported, others are not",
                "Fully supported by the evidence",
            ],
        },
    }


async def main(budget: float) -> None:
    _, val, _ = build_protocol_splits(200, 200)
    cache = CacheStore()
    client = JevDecisionsClient(cache_store=cache, concurrency_limit=8)
    if client.mock_mode:
        raise SystemExit("live mode required")
    http = client._ensure_loop_resources()
    sem = asyncio.Semaphore(8)
    spent = {"usd": 0.0, "calls": 0, "cached": 0, "refusals": 0, "fatal": None}
    rows = []

    async def one(inst):
        snippet, _ = retrieve_evidence(inst.evidence, inst.claim, 120, 20, 2)
        state = {"claim": inst.claim, "evidence": snippet}
        qs = questions_for(inst.claim)
        key = canonical_json_hash({"endpoint": client.endpoint, "mode": "live", "model": client.model,
                                   "state": state, "questions": qs, "exp": "multiq-v1"})
        hit = cache.get(key)
        if hit:
            raw, was_cached = hit[0], True
        else:
            if spent["fatal"] or spent["usd"] > budget:
                return
            async with sem:
                try:
                    raw = await client._call_single_decision_live(http, {"model": client.model, "state": state, "questions": qs})
                except FatalAPIError as e:
                    spent["fatal"] = str(e)
                    return
            was_cached = False
            cost = float(raw.get("usage", {}).get("cost", 0.0))
            spent["usd"] += cost
            spent["calls"] += 1
            cache.set(key=key, endpoint=client.endpoint, model=client.model, response_payload=raw,
                      latency_ms=0.0, cost_usd=cost)
        a = raw.get("answers", {})
        n, c, s = a.get("is_supported", {}), a.get("support_level", {}), a.get("support_score", {})
        if any(x.get("type") == "refusal" for x in (n, c, s)):
            spent["refusals"] += 1
        pr = c.get("probabilities") or {}
        sp = s.get("probabilities") or {}
        rows.append({
            "id": inst.id, "group": str(inst.metadata["group"]), "subdataset": inst.subdataset, "label": int(inst.label),
            "p_noul": n.get("noul"), "noul_conf": n.get("confidence"),
            "choice": c.get("choice"), "p_supported": pr.get("supported"), "p_unsupported": pr.get("unsupported"),
            "p_contradicted": pr.get("contradicted"), "choice_conf": c.get("confidence"),
            "score": s.get("score"), "score_conf": s.get("confidence"),
            **{f"score_p{i}": sp.get(str(i), sp.get(i)) for i in range(4)},
            "input_tokens": raw.get("usage", {}).get("input_tokens"), "cost": raw.get("usage", {}).get("cost"),
            "cached": was_cached, "model": raw.get("model"),
        })

    await asyncio.gather(*(one(i) for i in val))
    await client.aclose()
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "analysis" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "val_multiq.parquet", index=False)
    print(f"scored {len(df)}/{len(val)} claims; live calls={spent['calls']}, spent=${spent['usd']:.4f}, refusals={spent['refusals']}"
          + (f", ABORTED: {spent['fatal']}" if spent["fatal"] else ""))
    if len(df) < 50:
        return
    report(df)


def macro_auroc(df, col):
    vals = []
    for s, g in df.groupby("subdataset"):
        if g["label"].nunique() == 2 and g[col].notna().all():
            vals.append(auroc_score(g["label"].values, g[col].values))
    return float(np.mean(vals))


def report(df: pd.DataFrame) -> None:
    df = df.copy()
    df["score_norm"] = df["score"] / 3.0
    df["p_not_contra"] = 1.0 - df["p_contradicted"]
    y, g = df["label"].values.astype(int), df["group"].values
    print(f"\nfields present: noul_conf={df['noul_conf'].notna().mean():.0%}, choice_conf={df['choice_conf'].notna().mean():.0%}, "
          f"score_conf={df['score_conf'].notna().mean():.0%} | mean input tokens/call={df['input_tokens'].mean():.0f}, "
          f"mean cost/call=${df['cost'].astype(float).mean():.6f}")
    print("\n| signal | pooled AUROC [95% CI] | macro within-dataset AUROC | Brier (probability-like only) |\n|---|---|---|---|")
    for name, col, prob in [("noul P(true)  [baseline question]", "p_noul", True),
                            ("choice P(supported)", "p_supported", True),
                            ("choice 1-P(contradicted)", "p_not_contra", True),
                            ("score (probability-weighted, /3)", "score_norm", False)]:
        r = cluster_bootstrap_multi(g, y, df[col].values.astype(float), {"auroc": auroc_score}, 500, 0.95, 1337)["auroc"]
        b = f"{brier_score(y, df[col].values.astype(float)):.3f}" if prob else "n/a (not a probability)"
        print(f"| {name} | {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] | {macro_auroc(df, col):.3f} | {b} |")
    print("\nchoice argmax distribution:", df["choice"].value_counts().to_dict())
    print("corr(noul, choice P(supported)) = %.3f | corr(noul, score_norm) = %.3f" % (
        np.corrcoef(df["p_noul"], df["p_supported"])[0, 1], np.corrcoef(df["p_noul"], df["score_norm"])[0, 1]))
    # does the model's own `confidence` flag its mistakes? (choice argmax decision vs label)
    pred = (df["choice"] == "supported").astype(int).values
    wrong = (pred != y).astype(int)
    if len(set(wrong)) == 2:
        print("choice `confidence` as an error detector: AUROC(1-confidence -> wrong) = %.3f  (0.5 = useless)" %
              auroc_score(wrong, 1.0 - df["choice_conf"].astype(float).values))
    base = df["p_noul"].values
    print("\nsame question asked alone (baseline cache) vs inside the 3-question call: see compare step below")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=0.04)
    asyncio.run(main(ap.parse_args().budget_usd))
