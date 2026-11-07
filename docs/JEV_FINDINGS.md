# What we learned about Jev

A working write-up of how TypeSafe's Jev (`typesafe/jev-1.13` via the OpenRouter Decisions API) behaves when used
as a **claim verifier** (is a claim supported by an evidence text?). Everything marked *measured* comes from runs in
`results/runs/` and `results/analysis/`, on **LLM-AggreFact** (11 sub-datasets). Everything marked *docs* comes from the
OpenRouter / TypeSafe documentation. Unverified items are labelled as such.

## 1. What Jev is (docs)

Jev is a "decision model", not a chat model. You send `state` (text, and images from Jev 1.14) plus typed `questions`;
it returns typed answers with probabilities, no prose and no reasoning. Several questions can share one call and are
answered independently. Question types:

| Type | You supply | Answer fields |
|---|---|---|
| `noul` | `instructions`, optional `criteria: {true, false}` | `noul` = P(yes) |
| `choice` | `criteria`: option name -> guidance | `choice`, `probabilities`, `confidence` |
| `score` | `criteria`: ordered list of level descriptions | `score` (probability-weighted), `probabilities`, `legend`, `confidence` |
| `box` | `instructions` (images only, Jev 1.14+) | `box{top,bottom,left,right}`, `confidence` |

Other facts from the docs: input tokens cost $0.042 per million and output is free; default mode handles about 32 KB of
text (longer requests use a "slower and less accurate" long-context mode, up to 262k tokens); `refusal` is a possible
answer type; there is no `options` field (option names come from `criteria` keys); the endpoint is **alpha** and may change;
TypeSafe says probabilities differ between models, so thresholds must be re-picked per model.

Sources: [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev),
[What is Jev?](https://openrouter.ai/blog/insights/what-is-jev/),
[Decisions API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

## 2. How we used it

One `noul` question per (claim, evidence) pair, `state = {claim, evidence}`, evidence = top-2 BM25 snippets of 120 words from
the source document, one call per claim (`repeats=1`), pinned model `typesafe/jev-1.13` (the API resolved it to the dated
snapshot `typesafe/jev-1.13-20260917`). Output = raw P(supported), clipped to [0.01, 0.99].

## 3. Measured behaviour

**Cost and speed.** About 450-900 input tokens per call, about $0.00003 per call (matches the documented price exactly:
443 tokens x $0.042/M = $0.0000186), about 0.6 s latency, about 6 claims/s at 8 concurrent requests.

**Stability.** Two independent single calls differ by a mean of 0.0099 (p95 0.030, 0 of 100 decisions flipped). Repeating a
call 3x and averaging was statistically indistinguishable from one call (paired AUROC difference -0.0004), so we use one.
Asking `noul` inside a 3-question call changes it by only 0.0086 (within that noise floor): no cross-question interference.

**Prompt wording matters more than sampling noise.** Paraphrasing the instruction shifts probabilities by a mean 0.034
(p95 0.12, 1% decision flips). A more lenient wording shifted ExpertQA probabilities by about 0.11 and cut its raw Brier
0.292 -> 0.234 **without changing the ranking** (AUROC 0.649 -> 0.656).

**Output type.** `noul` is the best signal for this task (validation, n=738):

| Signal | Pooled AUROC [95% CI] | Macro within-dataset AUROC |
|---|---|---|
| `noul` P(true) | 0.870 [0.840, 0.895] | 0.803 |
| `choice` P(supported) | 0.858 [0.826, 0.885] | 0.798 |
| `score` (probability-weighted, /3) | 0.856 [0.821, 0.884] | 0.791 |
| `choice` 1 - P(contradicted) | 0.667 [0.628, 0.704] | 0.646 |

`choice` and `score` correlate 0.94 and 0.90 with `noul`, so they are largely redundant here; a 3-question call costs only
about 25% more than one (883 vs about 690 tokens). `noul` carried **no `confidence` field** on jev-1.13 (the blog is right; the
API reference page lists one). "Contradicted" is rarely chosen (36 of 738), because most unsupported claims in AggreFact are
unsupported but not contradicted.

**Uncertainty / abstention.** `noul`'s own margin `|P - 0.5|` is the best detector of its own mistakes (AUROC 0.735), a little
better than `choice` confidence (0.720) and `score` confidence (0.690). Keeping the most confident 80% of claims raises
decision accuracy from 0.782 to **0.839**, and 0.913 at 50% coverage.

**Evidence and context.** For datasets with long documents, more context helps; for short ones it makes no difference
(78% of ExpertQA and 96% of Reveal documents already fit in 2 snippets).

| Evidence (validation, n=738) | AUROC | Raw Brier | Macro within-dataset AUROC |
|---|---|---|---|
| 2 x 120-word snippets (baseline) | 0.870 | 0.155 | - |
| top-5 snippets | 0.878 | 0.141 | +0.017 |
| whole document (<= 32 KB for 99.1%) | 0.877 | 0.141 | +0.015 |

Gains concentrate in long-document datasets (TofuEval-MeetB +0.09 to +0.11, CNN +0.04 to +0.06, RAGTruth +0.02 to +0.04);
ClaimVerify gets slightly worse (-0.04). Pooled AUROC differences are not individually significant (p about 0.09-0.17).

**Discrimination is dataset-dependent.** Official test split, per-dataset AUROC from 0.62 (ExpertQA) to 0.97 (Reveal).
ExpertQA's weakness is **not** retrieval (the model already sees the whole document for 78% of its claims, and more or
different evidence changed nothing) and not wording (ranking unchanged); the cause is not established.

**Pooled metrics flatter Jev.** A predictor that ignores the claim and outputs each dataset's positive rate already scores
pooled AUROC 0.775 (Brier 0.179). Use within-dataset AUROC for claim-level discrimination.

## 4. Calibration

TypeSafe's calibration claim ("when it says 0.8, it is right about 80% of the time, averaged over many answers" - docs,
unverified by us) holds only roughly and only in aggregate. Test split, pooled: raw ECE 0.077; isotonic recalibration 0.036;
temperature scaling barely helps (0.075). Per dataset it is far from uniform (raw ECE up to about 0.3 on ExpertQA), and a
single global calibrator *worsens* ECE on the low-positive-rate datasets.

Out-of-fold on calibration+validation (document-grouped 5-fold CV): global isotonic Brier 0.146; logistic with length/BM25
features 0.144 (not significant after correction); **domain-aware** logistic with a per-dataset intercept **0.128**
[0.119, 0.137] and per-dataset isotonic 0.131. Domain-aware calibration needs the domain at inference time and labelled
calibration data for each new domain.

## 5. Pre-registered Gate 1 (test split, scored once)

AUROC 0.875 (pass, target >= 0.75), ECE 0.036 (pass, <= 0.08), nondeterminism 0.0099 (pass, <= 0.03), **Brier 0.138 (fail,
<= 0.12; CI [0.128, 0.147])**. Verdict as pre-registered: **FAIL**. Registration was committed (`8df7b6c`), but the code
tree was dirty when the test ran, so the exact code version is not pinned.

## 6. Practical guidance (from this evidence)

1. Use `noul` for support/entailment checks; extra question types add cost and little else for this task.
2. Pin the model version and record `resolved_version`; re-calibrate whenever it changes.
3. One call per query is enough; give the whole document when it is under about 32 KB (else a few BM25 snippets).
4. Calibrate **per domain** on a labelled sample if you will use the probability as a probability; do not trust pooled numbers.
5. For abstention, threshold on `|P - 0.5|`; expect about 0.84 accuracy at 80% coverage on this data mix.
6. Treat prompt wording as a parameter: changes of about 0.03 are routine, and wording can shift calibration without changing ranking.
7. Handle `refusal` answers and HTTP 402/429 explicitly; Jev gives no explanations, so log inputs for audit.

## 7. Not done / unverified

- **NLI baseline (running, results pending):** the same snippets scored by `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, to anchor
  whether Jev's numbers are good. See the section below once it finishes.
- No LLM-judge baseline; no comparison with the published LLM-AggreFact leaderboard.
- Jev 1.14 and `~typesafe/jev-latest` were not tested (version drift unknown); images/`box` out of scope.
- Domain-aware calibration was not re-run on top-5 / whole-document scores, and not confirmed on the test split.
- All results are one benchmark family, one pinned snapshot, and the free-tier allowance limited sample sizes
  (about 67 validation claims per dataset, 200 test).

## 8. Reproduce

```
python -m src.runs phase1 --spec configs/runs/phase1.yaml            # calibration + validation
python scripts/analyze_repeats.py                                     # repeats=1 vs 3 (offline)
python scripts/analyze_calibration.py                                 # calibration arms (offline)
python scripts/ablate_retrieval.py --subset val --variants top5,fulldoc,lenient
python scripts/explore_outputs.py                                     # noul + choice + score in one call
```

Total live spend for everything above: about $0.17 on the OpenRouter key.
