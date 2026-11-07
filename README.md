# Jev-Arbiter

Research code for **claim-level, calibration-weighted multi-agent arbitration** using the TypeSafe Jev
(`typesafe/jev-1.13`) Decisions API on OpenRouter as an atomic claim verifier.

> **Status (2026-11-07): Phase 1 (verifier calibration probe) is complete; the multi-agent arbiter is not yet
> evaluated.** Everything below marked *measured* comes from `results/runs/`. The multi-agent stages
> (clustering, fusion, baselines) exist as code but have **no validated results** - see `PLAN.md`.

**Start here for how Jev behaves (outputs, stability, calibration, practical guidance): [`docs/JEV_FINDINGS.md`](docs/JEV_FINDINGS.md).**

## What has been measured (Phase 1, LLM-AggreFact)

Jev as a single claim verifier, one call per claim, BM25-retrieved evidence, isotonic recalibration fitted on a
calibration split and selected on a validation split. Official **test** split (2,200 claims, 11 sub-datasets,
capped at 200 each), scored once after the thresholds were registered and committed (commit `8df7b6c`).
95% CIs are cluster-bootstrap (1,000 iterations) resampled by **source document**.

| Test split | Raw | Recalibrated (isotonic) | Pre-registered target | Result |
|---|---|---|---|---|
| AUROC | 0.877 [0.859, 0.892] | 0.875 [0.858, 0.891] | >= 0.75 | pass |
| ECE | 0.077 [0.064, 0.094] | 0.036 [0.029, 0.054] | <= 0.08 | pass |
| Brier | 0.145 [0.134, 0.157] | 0.138 [0.128, 0.147] | <= 0.12 | **fail** |
| Nondeterminism (mean abs diff) | - | 0.0099 | <= 0.03 | pass |

**Gate 1 verdict as pre-registered: FAIL** (the Brier target is missed and its CI excludes 0.12). Effective
independent votes N_eff = 1 (single verifier; not applicable at this stage).

Caveats you should not skip:
- Results depend on the sampling mix (equal cap per sub-dataset), and per-dataset behaviour varies widely
  (e.g. ExpertQA AUROC 0.62 [0.52, 0.72]; Reveal 0.97). A single global calibrator **improved** ECE on 6 of 11
  test datasets and **worsened** it on 5 (the low-positive-rate ones). Cause not yet established.
- The pre-registration is committed, but the working tree was dirty when the test ran, so the exact code
  version is not pinned by that commit (the run manifest records this).
- Jev's published figures (e.g. a 0.043 Brier) are **unverified** here and come from a different benchmark.
- Baselines: against one generic local NLI model on the same snippets (calibration+validation), Jev wins on all 11 datasets
  (domain-aware Brier 0.128 vs 0.159), and fusing the two does not beat Jev alone. There is no LLM-judge baseline and no
  comparison with specialised factual-consistency models, so this is not a state-of-the-art claim.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # OPENROUTER_API_KEY, HF_TOKEN (never commit .env)
pytest tests/                 # offline; 50+ tests, no network needed
```

LLM-AggreFact is a gated dataset (CC-BY-ND): accept its terms on Hugging Face with the account that owns
`HF_TOKEN`; files are downloaded to `data/aggrefact/` (git-ignored).

```bash
python -m src.runs status                      # is thresholds.yaml registered / committed?
python -m src.runs phase1 --mode mock          # free, offline pipeline check (mock verifier: not evidence)
python -m src.runs phase1                      # live, calibration + validation only (hard budget cap in the spec)
python -m src.runs register                    # write the pre-registration lock (then commit it)
python -m src.runs phase1 --include-test       # scores the test split; refused unless registration verifies
```

Each run writes `results/runs/<name>-<run_id>/` with `report.md`, `metrics.json`, `manifest.json` (code commit,
dirty flag, registration state, costs, error classes) and per-claim Parquet records.

## Design in one page

```
AggreFact -> document-grouped splits -> BM25 chunk retrieval -> Jev verifier (cached, pooled, rate-limited)
   hashed RunSpec        no doc in two splits     top-k snippets          raw P(supported)
   thresholds + lock  -> isotonic/temperature fit on calibration, selected on validation
   test split only with a verified registration -> cluster-bootstrap CIs -> Parquet + manifest + report
```

Invariants enforced in code (see `AGENTS.md`): content-addressed cache (mode and repeats in the key),
idempotent reruns with zero extra calls, seeds only from `configs/seeds.yaml`, billing/auth errors abort runs
loudly, no silent fallbacks, hard spend cap, and a registration lock for the test split.

## Repository layout

```
configs/   models.yaml, seeds.yaml, thresholds.yaml (+ lock), runs/phase1.yaml
src/config.py             typed config, seeds, pre-registration lock
src/clients/              Jev + LLM clients, SQLite content-addressed cache, mock
src/data/                 AggreFact loader, document-grouped splits, subsampling
src/retrieval/            chunking + BM25 evidence retrieval
src/verify/               Jev claim verifier
src/calibrate/            ECE / Brier / AUROC, temperature + isotonic scalers
src/eval/                 cluster-bootstrap CIs and paired tests
src/runs/                 RunSpec, Phase 1 pipeline, CLI (python -m src.runs)
src/claims, src/fuse/     multi-agent stages (NOT yet validated; fusion weights are placeholders)
scripts/legacy/           old keyword-label benchmark kept for reference only - do not use
app.py, run_cli.py        synthetic-fixture demos (not evidence)
tests/                    offline unit + end-to-end tests
```

## Roadmap

See `PLAN.md` (status table and amendments at the top). Next: baselines and a retrieval ablation for the
verifier (Stage 1b), then claim-level labels, fitted fusion and the Gate 2 comparison, then a distributed
verifier mesh.
