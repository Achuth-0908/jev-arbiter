# Jev-Arbiter

**A reproducible, pre-registered evaluation of [TypeSafe Jev](https://openrouter.ai/docs/guides/community/jev) as a claim verifier on LLM-AggreFact.**

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Status: research](https://img.shields.io/badge/status-research%20%2F%20Phase%201%20complete-orange.svg)

> **Independent project.** Not affiliated with, endorsed by, or sponsored by TypeSafe, OpenRouter, or the authors of
> LLM-AggreFact. Results are specific to one benchmark family and one pinned model snapshot.

## TL;DR

Jev is a "decision model": you send it text plus typed questions and it returns **probabilities**, not prose. This repo asks a
concrete question: *how well does Jev decide whether a claim is supported by its source document?* It answers with a leak-free,
pre-registered, fully reproducible evaluation (about **$0.00003 per claim**, about **$0.40** of API spend for the whole study).

| Pre-registered Gate 1, official **test** split (2,200 claims, 95% CIs) | Result | Target | |
|---|---|---|---|
| AUROC | 0.875 [0.858, 0.891] | >= 0.75 | pass |
| Expected calibration error (after recalibration) | 0.036 [0.029, 0.054] (raw 0.077) | <= 0.08 | pass |
| Brier score | 0.138 [0.128, 0.147] | <= 0.12 | **fail** |
| Run-to-run difference between independent calls | 0.0099 | <= 0.03 | pass |

**Verdict as pre-registered: FAIL (Brier).** The failure is reported, not amended away; the diagnosis is the interesting part
(see [Key findings](#key-findings)).

## Contents
[What is Jev?](#what-is-jev) | [Key findings](#key-findings) | [How it works](#how-it-works) | [Getting started](#getting-started) |
[Reproducing the analyses](#reproducing-the-analyses) | [Methodology](#methodology) | [Repository layout](#repository-layout) |
[Limitations](#limitations) | [Roadmap](#roadmap) | [Licensing and attribution](#licensing-and-attribution) |
[Citation](#citation) | [Contributing](#contributing)

## What is Jev?

Jev is TypeSafe's decision model, exposed through the OpenRouter **Decisions API** (alpha: the format may change). It does not
generate text or show reasoning. You send `state` (text) and a map of typed `questions`; each is answered independently, and
several can share one call.

| Question type | You supply | Returns |
|---|---|---|
| `noul` | instructions, optional `criteria: {true, false}` | `noul` = P(yes) |
| `choice` | `criteria`: option name -> guidance | the choice, per-option probabilities, `confidence` |
| `score` | `criteria`: ordered list of levels | probability-weighted `score`, distribution, `confidence` |
| `box` | instructions (images only, Jev 1.14+) | a bounding box and `confidence` (not tested here) |

Pricing is per input token (output is free); at the time of writing that is $0.042 per million input tokens. Primary sources:
[Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev),
[What is Jev?](https://openrouter.ai/blog/insights/what-is-jev/),
[Decisions API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

This project uses one `noul` question: *"Is the claim strictly supported by the evidence?"* with `state = {claim, evidence}`.

## Key findings

Only the table in the TL;DR is a **confirmatory** result (test split, scored once, after the protocol was committed). Everything
below was measured on the calibration and validation splits and is **exploratory**. Full detail and caveats:
[`docs/JEV_FINDINGS.md`](docs/JEV_FINDINGS.md).

1. **Pooled AUROC flatters.** A predictor that ignores the claim and outputs each dataset's base rate scores pooled AUROC 0.775
   (Brier 0.179). Within a dataset, Jev's raw AUROC on the test split ranges from **0.62 (ExpertQA) to 0.97 (Reveal)**.
2. **Calibration is domain-dependent.** A single global calibrator improved ECE on some datasets and worsened it on others.
   Adding a per-dataset term cut Brier from 0.146 to **0.128** (document-grouped 5-fold CV). It needs the domain at inference time.
3. **Gate 1 failed on Brier, and recalibration cannot fix that.** Brier is limited by ranking quality, not just calibration.
4. **Jev is cheap and stable.** About $0.00003 and about 0.6 s per call, or about 6 claims/s at 8 concurrent requests.
   Independent calls differ by about 0.01 in probability; averaging 3 calls was statistically indistinguishable from 1 (paired AUROC difference -0.0004).
5. **Prompt wording moves probabilities more than sampling noise** (about 0.034 vs about 0.01), and can shift calibration
   without changing the ranking.
6. **More context helps only on long documents.** Passing the whole document (99% of validation documents fit Jev's default 32 KB window) raised
   validation Brier from 0.155 to 0.141, mostly on long-document datasets. ExpertQA's weakness is *not* a retrieval problem.
7. **`noul` is the best signal.** `choice` and `score` correlate 0.94 and 0.90 with it and are not better. Its own margin
   `|P - 0.5|` works for abstention: keeping the most confident 80% of claims lifts decision accuracy from 0.78 to 0.84.
8. **Against a generic local NLI model** (DeBERTa-v3-base MNLI/FEVER/ANLI, same snippets), Jev scored higher on all 11 datasets
   (domain-aware Brier 0.128 vs 0.159). **Fusing the two added nothing**: the fitted weight on NLI was about zero, even though the
   measured error correlation was only 0.41 (so a vote-counting "effective votes" argument would have overstated the gain).

## How it works

```
LLM-AggreFact (gated)  ->  document-grouped splits  ->  BM25 evidence retrieval  ->  Jev verifier
 official dev / test       cal / val from dev;          chunk + top-k snippets      cached, pooled async
                           test = official test;        (or the whole document)     client, spend cap
                           no document in 2 splits
                                                                  |
   configs/thresholds.yaml + lock  ->  calibrators fit on calibration, selected on validation (isotonic / temperature)
                                                                  |
   test split scored ONLY with a verified registration  ->  cluster-bootstrap CIs  ->  Parquet + manifest + report
```

Engineering principles, each enforced in code and covered by offline tests (see [`AGENTS.md`](AGENTS.md)):

- **Leak-free evaluation.** Splits keep each source document in exactly one split. The official dev and test files share some
  documents (about 36% of one dataset's test claims), so those are removed from the calibration pool.
- **Pre-registration with teeth.** Targets and protocol live in a committed `configs/thresholds.yaml`, hashed into a lock. The
  runner refuses to score the test split unless the hash matches; the manifest records whether the registration is committed.
- **Content-addressed cache.** Every external call is keyed by (endpoint, mode, model, state, questions, repeats). Re-running an
  experiment makes **zero** API calls. Mock and live results can never be mixed.
- **No silent failures.** Unrecognised payloads raise; auth/billing errors (401/402/403) abort the run; a live run refuses to
  fall back to mock; a hard `budget_usd` cap stops a run cleanly before it overspends.
- **Statistics done properly.** Cluster-bootstrap CIs resampled by source document (>= 1,000 iterations), reported next to every
  headline number, with raw and recalibrated ECE and Brier.
- **Run manifests.** Each run writes code commit, dirty flag, resolved model version, costs, error classes and registration state.
- **Speed.** A pooled async client with bounded concurrency scores claims at about 4x the throughput of a sequential loop.

## Getting started

### Prerequisites
- Python 3.10+ (developed and tested on 3.11, Windows)
- For live runs: an [OpenRouter API key](https://openrouter.ai/keys) with a few cents of credit or free-tier allowance
- For the dataset: a Hugging Face token, plus acceptance of the dataset terms (below)

### Install
```bash
git clone https://github.com/Achuth-0908/jev-arbiter.git
cd jev-arbiter
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Optional extras: `pip install "streamlit>=1.30"` (demo app), `pip install torch "transformers>=4.40"` (local NLI baseline).

### Configure
```bash
cp .env.example .env      # Windows: copy .env.example .env
```
| Variable | Needed for | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | live Jev calls | never committed; `.env` is git-ignored |
| `HF_TOKEN` | downloading LLM-AggreFact | the account must have accepted the dataset terms |
| `DECISIONS_API_URL`, `DECISIONS_MODEL` | overrides | defaults: the OpenRouter alpha endpoint, `typesafe/jev-1.13` |
| `MOCK_API_MODE` | forcing the offline mock | mock scores are **not evidence** |

### Run the tests (no network, no keys, no dataset)
```bash
pytest tests/
```

### Get the dataset
[LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) is gated (CC-BY-ND-4.0, evaluation use). Accept its terms on
Hugging Face with the account that owns `HF_TOKEN`; the loader downloads it to `data/aggrefact/` (git-ignored) on first use.
**Every `phase1` command below needs this**, including mock mode.

### Run Phase 1
```bash
python -m src.runs status                      # is configs/thresholds.yaml registered / committed?
python -m src.runs phase1                      # calibration + validation only (live; budget cap in configs/runs/phase1.yaml)
python -m src.runs phase1 --mode mock          # pipeline check with the mock verifier (still needs the dataset; not evidence)
python -m src.runs register                    # write the pre-registration lock (then commit thresholds.yaml + the lock)
python -m src.runs phase1 --include-test       # scores the TEST split; refused unless the registration verifies
python -m src.runs phase1 --budget-usd 0.05    # override the spend cap
```
Each run writes `results/runs/<name>-<run_id>/` with `report.md`, `metrics.json`, `manifest.json` and per-claim Parquet records.
That folder is git-ignored because it derives from a licensed dataset.

### Demo app
```bash
pip install "streamlit>=1.30" && streamlit run app.py
```
The **Phase 1 Results** and **Calibration & Reliability** tabs read the latest run in `results/runs/`. The playground tab uses a tiny
synthetic fixture and the mock verifier and is a demo only.

## Reproducing the analyses

The cache is not in the repo, so reproducing means re-running the live calls once (roughly $0.13 for the headline Phase 1 numbers
(4,400 claims at one call each); the exploratory scripts add a few cents each). After that, everything replays from the local cache for free.

| Script | What it answers | Needs |
|---|---|---|
| `scripts/analyze_repeats.py` | Is 1 call as good as 3? | cached 3-repeat calls from the original `repeats: 3` run (the current spec uses 1) |
| `scripts/analyze_calibration.py` | Do domain-aware calibrators beat a global one? | a Phase 1 run |
| `scripts/ablate_retrieval.py` | Is a weak dataset a retrieval problem? (top-5, bigger chunks, whole document, wording) | live calls, capped |
| `scripts/explore_outputs.py` | How do `noul`, `choice` and `score` compare in one call? | live calls, capped |
| `scripts/export_nli_inputs.py`, `scripts/nli_score.py`, `scripts/compare_nli.py` | Jev vs a local NLI baseline | `torch`, `transformers`, about 1 CPU-hour |
| `scripts/fuse_verifiers.py` | Does fusing Jev with NLI help? | the NLI scores |

`scripts/legacy/` holds an early prototype benchmark with hardcoded labels, kept only as a warning. Do not use it.

## Methodology

- **Task and data.** Binary claim verification (supported or not) on LLM-AggreFact: 11 sub-datasets, 30,420 dev and 29,320 test
  claims. Each sub-dataset is capped (seeded) at 200 claims from dev and 200 from test so the largest does not dominate.
- **Splits.** Calibration (about 1,460) and validation (about 740) come from the official dev split, divided by source document;
  the test split (2,200) is the official test split. Seeds come from `configs/seeds.yaml`.
- **Evidence.** Top-2 BM25 snippets of 120 words per claim for the confirmatory run; whole-document and top-5 variants were
  explored on validation only.
- **Calibration.** Temperature scaling and isotonic regression are fit on the calibration split; the calibrator is chosen by
  validation ECE (a pre-registered rule), never on test.
- **Metrics.** AUROC, ECE (10 bins), Brier, accuracy, with 1,000-iteration cluster-bootstrap 95% CIs grouped by source document.
- **Deviations from the original plan** are logged in [`AGENTS.md`](AGENTS.md) section 5 and in the amendments at the top of
  [`PLAN.md`](PLAN.md), including the Brier target, `repeats: 3 -> 1`, and replacing an invalid option-swap probe.

## Repository layout

```
configs/        models.yaml, seeds.yaml, thresholds.yaml (+ lock), runs/phase1.yaml
docs/           JEV_FINDINGS.md  - the full write-up of what was measured
src/
  config.py     typed config, seeds, pre-registration lock
  clients/      Jev + LLM clients, SQLite content-addressed cache, offline mock
  data/         AggreFact loader, document-grouped splits, seeded subsampling
  retrieval/    chunking and BM25 evidence retrieval
  verify/       Jev claim verifier
  calibrate/    ECE / Brier / AUROC, temperature and isotonic scalers
  eval/         cluster-bootstrap CIs and paired tests
  runs/         RunSpec, Phase 1 pipeline, CLI (python -m src.runs)
  claims/, fuse/   multi-agent stages: EXPERIMENTAL, not validated (see Limitations)
scripts/        analysis and exploration scripts (see table above); legacy/ is a warning
tests/          offline unit and end-to-end tests
app.py          Streamlit demo reading real run artifacts
AGENTS.md       engineering and scientific invariants       PLAN.md  original plan with amendments
```

## Limitations

Read these before citing any number here.

- **One benchmark family, one pinned model snapshot** (`typesafe/jev-1.13`, resolved to a dated build), small samples per dataset
  (about 67 validation and 200 test claims each). The Decisions API is alpha.
- **Confirmatory vs exploratory.** Only raw Jev with isotonic recalibration was evaluated on the untouched test split. Domain-aware
  calibration, evidence variants, the NLI comparison and fusion are exploratory, on calibration and validation data.
- **No LLM-judge baseline.** This project does **not** show Jev is better or cheaper than an LLM-as-a-judge. The only baseline is
  one generic NLI model; specialised factual-consistency models are likely stronger and were not tested.
- **Pooled metrics depend on the sampling mix** (equal cap per dataset); per-dataset results vary widely.
- **The Gate 1 test run executed from a working tree with uncommitted changes.** The pre-registration itself was committed first
  (`8df7b6c`). Afterwards the code was committed (`881e2ee`); no `src/` or `configs/` file was modified after the test run
  finished (checked by file modification time), so that commit contains the code that produced the result. The run manifest
  records `dirty: true`.
- **The multi-agent arbitration code is experimental.** `src/fuse/` and `src/claims/` contain unvalidated placeholder weights and an
  N_eff discount that this work showed is incomplete (it ignores differences in verifier accuracy). The multi-agent thesis in
  `PLAN.md` has **not** been evaluated.
- Jev figures quoted in `PLAN.md` from earlier sources (for example a 0.043 Brier) are **unverified** and come from a different setting.

## Roadmap

1. LLM-judge baseline under the same protocol and a registered comparison.
2. Confirmatory test of domain-aware calibration (a new registered protocol).
3. Verify-and-abstain pipeline: generate with an LLM, verify claims with Jev, abstain on weak ones.
4. Prompt-injection robustness and a cost/latency frontier.
5. Longer term: pluggable verifiers behind one interface and a distributed verification layer.

## Licensing and attribution

- **Code:** [MIT](LICENSE).
- **Data:** LLM-AggreFact is licensed **CC-BY-ND-4.0** and its card states it is for **evaluation only** and should not be used to
  pretrain or fine-tune NLP models. This repo does not redistribute it (`data/` is git-ignored; only aggregate statistics are
  published), trains no NLP model on it, and fits only small calibrators (a few parameters) on the dev split. Read the dataset
  terms yourself before using it. Please cite the dataset authors (below).
- **Models and services:** the NLI baseline is [`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
  (MIT). Jev is a product of TypeSafe, accessed through OpenRouter under their terms.

## Citation

If you use this repository, please cite it (and the dataset it evaluates on):

```bibtex
@software{jev_arbiter_2026,
  author = {G Achuth},
  title  = {Jev-Arbiter: a reproducible, pre-registered evaluation of Jev as a claim verifier},
  year   = {2026},
  url    = {https://github.com/Achuth-0908/jev-arbiter}
}

@inproceedings{tang-etal-2024-minicheck,
  title     = {MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents},
  author    = {Tang, Liyan and Laban, Philippe and Durrett, Greg},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024}
}
```
(The dataset card on Hugging Face has the canonical BibTeX entry for LLM-AggreFact.)

## Contributing

Issues and pull requests are welcome. Please keep the invariants in [`AGENTS.md`](AGENTS.md): no secrets in commits, every external
call through the cache, seeds from `configs/seeds.yaml`, the test split untouched during development, and an offline test suite
that passes with `pytest tests/`. If you find a credential in this repository's history, please open a private security advisory
instead of a public issue.
