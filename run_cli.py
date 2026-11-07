"""Command-line interface for jev-arbiter evaluation, calibration probes, and arbitration."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from src.clients.cache import CacheStore
from src.clients.jev import JevDecisionsClient
from src.data.loaders import generate_benchmark_synthetic_subset, create_frozen_splits
from src.data.qa_loader import get_sample_multihop_instances
from src.eval.runner import BenchmarkRunner
from src.fuse.engine import ArbitrationEngine
from src.verify.verifier import JevClaimVerifier


async def run_probe(is_mock: bool = True):
    print("\n" + "=" * 70)
    print("PHASE 1: JEV CALIBRATION PROBE (SYNTHETIC DEMO FIXTURE - NOT EVIDENCE)")
    print("=" * 70)
    print("For real, leakage-safe, pre-registered results run: python -m src.runs phase1")

    cache = CacheStore()
    client = JevDecisionsClient(mock_mode=is_mock, cache_store=cache)
    verifier = JevClaimVerifier(client=client)
    runner = BenchmarkRunner(verifier)

    instances = generate_benchmark_synthetic_subset(n_samples=120, seed=42)
    cal_set, val_set, test_set = create_frozen_splits(instances, cal_ratio=0.40, val_ratio=0.20, test_ratio=0.40, seed=42)

    print(f"Loaded {len(instances)} instances:")
    print(f"  - Calibration split: {len(cal_set)} instances")
    print(f"  - Validation split:  {len(val_set)} instances")
    print(f"  - Test split:        {len(test_set)} instances (Held-out)")
    print(f"  - Mode:              {'Offline Mock Mode ($0 cost)' if is_mock else 'Live OpenRouter Decisions API'}")
    print("\nFitting recalibration scalers and evaluating test split...")

    results = await runner.run_phase1_calibration_probe(cal_set, test_set)

    raw = results["raw_jev"]
    temp = results["temperature_scaled"]
    iso = results["isotonic_scaled"]

    print("\n--- RESULTS ON HELD-OUT TEST SPLIT ---")
    print(f"Fitted Temperature: T = {results['fitted_temperature']:.3f}\n")
    print(f"{'Metric':<18} | {'Raw Jev-1.13':<14} | {'Temp Scaled':<14} | {'Isotonic Scaled':<14}")
    print("-" * 68)
    print(f"{'ECE (lower=better)':<18} | {raw['ece']:<14.4f} | {temp['ece']:<14.4f} | {iso['ece']:<14.4f}")
    print(f"{'Brier (lower=better)':<18} | {raw['brier']:<14.4f} | {temp['brier']:<14.4f} | {iso['brier']:<14.4f}")
    print(f"{'AUROC (higher=better)':<18} | {raw['auroc']:<14.4f} | {temp['auroc']:<14.4f} | {iso['auroc']:<14.4f}")
    print(f"{'Accuracy':<18} | {raw['accuracy']*100:<13.1f}% | {temp['accuracy']*100:<13.1f}% | {iso['accuracy']*100:<13.1f}%")
    print("-" * 68)
    print(f"Mean repeat non-determinism variance: {results['mean_repeat_variance']:.5f}")
    print(f"Mean option-invariance divergence:     {results['mean_option_invariance_divergence']:.5f}")


async def run_arbitration(
    strategy: str = "calibrated_fusion",
    is_mock: bool = True,
    example_idx: int = 0,
):
    print("\n" + "=" * 70)
    has_jev = strategy in ("raw_jev", "calibrated_jev", "calibrated_fusion")
    jev_label = f"WITH Jev ({'Mock' if is_mock else 'Live OpenRouter'})" if has_jev else "WITHOUT Jev (Baseline Multi-Agent Voting)"
    print(f"ARBITRATION RUNNER: [{strategy.upper()}] — {jev_label}")
    print("=" * 70)

    samples = get_sample_multihop_instances()
    qa = samples[example_idx % len(samples)]
    print(f"Query: {qa.question}\n")

    client = JevDecisionsClient(mock_mode=is_mock)
    verifier = JevClaimVerifier(client=client)
    engine = ArbitrationEngine(verifier=verifier)

    res = await engine.arbitrate(
        query=qa.question,
        candidate_responses=qa.candidate_responses,
        candidate_confidences=qa.candidate_confidences,
        reference_documents=qa.passages,
        strategy=strategy,
    )

    print(f"{'Sub-Claim Proposition':<48} | {'Votes':<5} | {'Eff.Ag':<7} | {'Jev P':<6} | {'Fused P':<8} | {'Status':<10}")
    print("-" * 94)
    for ac in res.arbitrated_claims:
        print(f"{ac.canonical_claim[:48]:<48} | {ac.agreement_count}/4   | {ac.effective_agreement_ratio:<7.2f} | {ac.calibrated_jev_prob:<6.2f} | {ac.fused_prob:<8.2f} | {ac.status:<10}")

    print("\n" + "-" * 70)
    print("FINAL ASSEMBLED ANSWER (Exclusively from verified claims):")
    print(res.final_answer)
    print("-" * 70)
    print(f"Latency: {res.total_latency_ms:.1f}ms | Cost: ${res.total_cost_usd:.5f} | N_eff: {res.effective_independent_votes:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Jev-Arbiter CLI Runner")
    parser.add_argument("command", choices=["probe", "arbitrate"], default="arbitrate", nargs="?")
    parser.add_argument(
        "--strategy",
        choices=["majority_vote", "self_confidence", "raw_jev", "calibrated_jev", "calibrated_fusion"],
        default="calibrated_fusion",
        help="Arbitration strategy: majority_vote / self_confidence (WITHOUT Jev), or raw_jev / calibrated_jev / calibrated_fusion (WITH Jev)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute against live OpenRouter Decisions API (requires OPENROUTER_API_KEY). Default is offline mock mode.",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=0,
        help="Example index from HotpotQA multi-hop benchmark (0, 1, or 2)",
    )
    args = parser.parse_args()

    is_mock = not args.live

    if args.command == "probe":
        asyncio.run(run_probe(is_mock=is_mock))
    elif args.command == "arbitrate":
        asyncio.run(run_arbitration(strategy=args.strategy, is_mock=is_mock, example_idx=args.example))


if __name__ == "__main__":
    main()
