# Calibration-Weighted Multi-Agent Arbitration with Jev (`jev-arbiter`)

> ## Status & Amendments (2026-11-07)
> This plan was written before any measurement. What has actually been done, and what changed:
>
> | Phase | Status |
> |---|---|
> | 0 Foundations | Done: content-addressed cache (mode/repeats in key), pooled async client, hashed run specs, pre-registration lock, manifests. |
> | 1 Jev probe (Gate 1) | Calibration + validation done on real LLM-AggreFact with document-grouped splits and cluster-bootstrap CIs. Test split scored only after registration (see `results/runs/*/report.md`). |
> | 1b Anchor & diagnose | **New, not started:** LLM-judge / NLI baselines on the same splits, retrieval ablation, dataset-conditioned calibration. |
> | 2 Multi-agent baselines | Not started (needs claim-level labels and candidate agents). |
> | 3 Fusion / Gate 2 | Not started. The weights currently in `src/fuse/engine.py` are invented placeholders, not fitted. |
> | 4 Robustness / cost frontier | Not started. The regex injection filter is unmeasured. |
> | 5 Demo / write-up | Not started. `app.py` and `run_cli.py` run on a synthetic fixture and are demo-only. |
> | 6 Distributed verifier mesh | **New, long-term goal.** Plan: pluggable verifiers, queue + stateless workers over the shared cache, then hedging/quorum/fault-injection experiments (developed first against simulated verifiers at $0). |
>
> **Amendments (decided on calibration/validation data only, before the test split was scored):**
> 1. *Gate 1 had two definitions.* This plan lists ECE and AUROC; `configs/thresholds.yaml` also has a Brier target (0.12). The YAML is authoritative and was kept unchanged. Brier on validation is about 0.14, so that criterion is expected to fail and will be reported as such.
> 2. *Option-name invariance probe replaced.* The live Decisions API takes no `options`, so swapping them sent an identical request and measured nothing. It is replaced by a **wording-invariance** probe (paraphrased instructions).
> 3. *`repeats` 3 -> 1.* Repeat-1 vs mean-of-3: paired AUROC difference -0.0004, 0.9% decision flips, about one third of the cost. Registered in `thresholds.yaml` (`phase1.repeats`).
> 4. *Unverified figures.* The Jev statistics quoted below (0.043 Brier, ~1.3% rerun divergence) are carried over from an earlier source and have **not** been verified. Measured on our data: rerun divergence about 0.006 mean |dp| (repeat-3) and the figures in the run reports.
> 5. *Pooled metrics depend on the sampling mix.* Splits cap each sub-dataset at 200 claims, so pooled ECE/Brier reflect that mixture, and per-dataset results differ widely (e.g. ExpertQA AUROC near chance on validation; cause not yet established, retrieval is a suspect).
> 6. *Not formally committed beforehand unless stated in the run manifest.* See `AGENTS.md` section 5.

## 1. Executive Summary & Research Gap

### The Core Idea
Standard multi-agent aggregation relies on majority voting (self-consistency) or naive confidence averaging. However:
- **Self-reported LLM confidence** is notoriously miscalibrated and vendor-biased (often collapsing to 0 or 1 with no meaningful middle band).
- **Multi-agent debate** frequently degenerates due to sycophancy or cascading errors among correlated agents.
- **Correlated errors** cap information gain: an ensemble of $k$ agents with average pairwise error correlation $\bar{\phi}$ yields only $N_{\text{eff}} = \frac{k}{1 + (k-1)\bar{\phi}}$ independent votes.

### The Specific Novelty & Gap
While confidence-weighted voting and atomic claim reassembly exist independently:
**No existing work evaluates an external typed decision model (TypeSafe Jev `typesafe/jev-1.13` via Decisions API) as an atomic per-claim verifier across a heterogeneous multi-agent pool, coupled with held-out recalibration, agent error-correlation discounting, and a rigorous cost-accuracy frontier.**

Recent evaluations are reported to show Jev achieving competitive accuracy and superior Brier scores (e.g. 0.043 vs 0.054 against frontier LLM judges) with non-collapsed probability distributions across the $[0.01, 0.99]$ grid. **[UNVERIFIED: carried over from an earlier source; our own measurements are in `results/runs/`. On our pooled data about 27% of probabilities were below 0.05 or above 0.95.]** However, Jev is vulnerable to:
1. Workload-specific calibration shift (requiring empirical post-scaling).
2. Prompt injection and candidate text context rot (requiring decontextualized atomic extraction and BM25 snippet grounding).
3. Minor nondeterminism (~1.3% rerun divergence) and option-name sensitivity.

---

## 2. System Architecture

```
                    ┌────────────────────────────┐
                    │       User Query / QA      │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
  │ Candidate 1 │          │ Candidate 2 │          │ Candidate 3 │
  │  (Claude)   │          │  (Llama-3)  │          │  (Gemini)   │
  └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Atomic Claim Extraction   │
                    │   & Sanitization (Cheap)   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Claim Normalization        │
                    │ & Semantic Clustering      │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ BM25 Snippet Retrieval     │
                    │ (Small, Clean Context)     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Jev Verification (Noul)    │
                    │ (3x Repetitions, Averaged) │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Domain Recalibration Layer │
                    │ (Isotonic / Platt Scaling) │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Calibrated Fusion Engine   │
                    │ - Calibrated P(Jev)        │
                    │ - Agreement (N_eff scaled) │
                    │ - BM25 Score & Self-Conf   │
                    └─────────────┬──────────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
             [P >= Tau_accept]         [P < Tau_accept]
           Accepted Atomic Claims    Abstain / Escalate
                     │
                     ▼
       ┌───────────────────────────┐
       │     Final Answer Writer   │
       │  (Only Accepted Claims)   │
       └───────────────────────────┘
```

### Component Details
1. **Candidate Agents:** 3+ heterogeneous model families (e.g., Anthropic Claude, Meta Llama-3, Google Gemini, Mistral) with varied system prompts.
2. **Claim Extraction & Sanitization:** A lightweight model breaks agent outputs into atomic, decontextualized factual claims. This step acts as an injection firewall by preventing direct pass-through of untrusted raw candidate prose.
3. **Claim Clustering:** Group semantically identical sub-claims across candidates, tracking candidate support and agent self-reported confidence.
4. **BM25 Retrieval:** Fetch 1-3 targeted snippets per claim to prevent context rot and keep state minimal.
5. **Jev Verification (`typesafe/jev-1.13` Decisions API):**
   - Formatted as a **Noul** decision problem per (claim, evidence).
   - Base URL explicitly configured to `openrouter.ai/api/alpha/decisions`.
   - Repeated 3 times and averaged to mitigate ~1.3% nondeterminism, clipped to $[0.01, 0.99]$.
6. **Recalibration:** Temperature scaling or isotonic regression fitted *strictly* on held-out calibration splits.
7. **Fusion Model:** Cross-fitted logistic regression combining:
   - Recalibrated Jev probability $P_{\text{cal}}(\text{Jev})$
   - Agent agreement count discounted by effective independent votes: $N_{\text{eff}} = \frac{k}{1 + (k-1)\bar{\phi}}$
   - Agent self-reported confidence
   - BM25 retrieval score
8. **Decision & Reassembly:** Apply frozen threshold $\tau$. If accepted, a synthesis model compiles the final answer exclusively from verified claims.

---

## 3. Mathematical Foundations

### 3.1 Effective Independent Votes
When $k$ agents make judgments, their pairwise error correlation $\bar{\phi}$ reduces information capacity:
$$N_{\text{eff}} = \frac{k}{1 + (k - 1) \bar{\phi}}$$
where $\bar{\phi} = \frac{2}{k(k-1)} \sum_{i < j} \phi_{ij}$, and $\phi_{ij}$ is the Matthews Correlation Coefficient or Pearson correlation between error residuals of agent $i$ and agent $j$.

### 3.2 Calibration Metrics
- **Brier Score:**
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
- **AUROC:** Evaluates ranking capability of probabilities independent of decision threshold.

### 3.3 Recalibration Models
1. **Temperature Scaling (Platt Scaling):**
   $$p_{\text{cal}} = \sigma\left(\frac{\text{logit}(p)}{T}\right)$$
   where $T > 0$ is optimized via NLL on held-out calibration set.
2. **Isotonic Regression:** Non-parametric monotonic step function minimizing mean squared error on calibration split.

---

## 4. Experimental Phases & Gates

### Phase 0: Foundations & Cached Clients (Days 1-2)
- **Objective:** Build robust client abstraction, SQLite/JSON content-addressed cache, and record/replay mock.
- **Cache Key:** SHA256 of `(endpoint, model, json_canonical(state), json_canonical(questions))`.
- **Target:** 100 repeated queries execute with 0 network calls; test suite passes offline.

### Phase 1: Jev Calibration Probe (Days 3-4) — **Gate 1**
- **Objective:** Evaluate `typesafe/jev-1.13` standalone calibration on LLM-AggreFact subset (claims checked against grounding docs).
- **Split:** 40% calibration, 20% validation (threshold freezing), 40% test.
- **Protocol:** One Noul query per claim (`repeats: 1`, see Amendments). Compute raw vs recalibrated ECE, Brier, AUROC with 1,000-iteration cluster-bootstrap CIs grouped by source document.
- **Robustness checks:** Fresh-call nondeterminism and *wording* invariance (paraphrased instructions). (Option-name swap was dropped: the API takes no `options`.)
- **Gate 1 Criterion (as pre-registered in `configs/thresholds.yaml`):** post-calibration AUROC $\ge 0.75$, ECE $\le 0.08$, Brier $\le 0.12$, nondeterminism divergence $\le 0.03$; calibrator chosen on validation only; evaluated once on the untouched test split. If failed, pivot Jev to selective routing/abstention only.

### Phase 2: Multi-Agent Candidate Pipeline & Baselines (Week 2)
- **Objective:** End-to-end multi-agent pipeline on HotpotQA (distractor setting).
- **Baselines:**
  1. Single Best Agent
  2. Majority Vote (unweighted)
  3. Self-Confidence Weighted Vote
  4. LLM Aggregator (reading full reasoning traces)
  5. Draft-Verify Cascade (cheap draft + frontier verify)
- **Metrics:** Factual Precision, Recall, F1, Selective Accuracy vs Coverage curve.
- **Bootstrap:** Cluster-bootstrap 95% CIs resampled at the *question* level.

### Phase 3: Jev-Verified Fusion & Correlation Analysis (Week 3) — **Gate 2**
- **Objective:** Train and evaluate fusion model incorporating $N_{\text{eff}}$ discounted vote and calibrated Jev probabilities.
- **Key Ablation:** Does calibrated Jev verification provide statistically significant gain over vote count + agent agreement after accounting for error correlation $\bar{\phi}$?
- **Gate 2 Criterion:** Statistically significant improvement ($p < 0.05$ via cluster bootstrap) over majority vote and self-confidence weighting. If not met, reframe as rigorous negative-result publication on correlation limits.

### Phase 4: Robustness & Cost-Accuracy Frontier (Days 4-5)
- **Prompt Injection Resilience:** Measure decision shift when adversarial instructions are embedded in candidate text.
- **Cost Frontier:** Latency (ms) and API Cost ($) vs Accuracy across all architectures with and without cache.
- **Transfer Check:** Evaluate frozen thresholds on out-of-distribution dataset.

### Phase 5: Demo & Write-Up (Week 4)
- Comprehensive research report with reliability diagrams, Pareto frontier plots, and ablation tables.
- Interactive Gradio/Streamlit arbitration demo.

---

## 5. Repository Structure

```
jev-arbiter/
├── PLAN.md                  # This master research and implementation plan
├── AGENTS.md                # Agent guidelines, safety & invariants
├── pyproject.toml           # Package specification & dependencies
├── .env.example             # Template for API keys
├── .gitignore               # Strict exclusion of keys, caches, artifacts
├── configs/
│   ├── models.yaml          # Model endpoints, families, temperatures
│   ├── seeds.yaml           # Fixed seeds for deterministic splitting
│   └── thresholds.yaml      # Pre-registered thresholds & clip bounds
├── src/
│   ├── clients/             # Jev Decisions API & LLM clients with caching & mock
│   ├── data/                # Dataset loaders (LLM-AggreFact, HotpotQA, splits)
│   ├── claims/              # Atomic claim extraction, sanitizer, clustering
│   ├── retrieval/           # BM25 evidence indexing & retrieval
│   ├── verify/              # Jev Noul prompts, multi-run aggregation
│   ├── calibrate/           # Platt, Isotonic scaling, ECE, Brier, AUROC
│   ├── fuse/                # Effective votes, logistic fusion, arbitration
│   └── eval/                # Bootstrap CI, metrics runner, plotting
├── tests/                   # Unit & integration tests (record/replay mock)
├── results/                 # Experiment logs, metrics, figures
└── notebooks/               # Exploratory data analysis & figure reproduction
```

---

## 6. Safety & Key Management Rules
- **NEVER** log, print, or commit API keys.
- All secrets are loaded from `.env` via `python-dotenv`.
- Offline mock mode is default for unit tests and local development.
