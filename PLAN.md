# Calibration-Weighted Multi-Agent Arbitration with Jev (`jev-arbiter`)

## 1. Executive Summary & Research Gap

### The Core Idea
Standard multi-agent aggregation relies on majority voting (self-consistency) or naive confidence averaging. However:
- **Self-reported LLM confidence** is notoriously miscalibrated and vendor-biased (often collapsing to 0 or 1 with no meaningful middle band).
- **Multi-agent debate** frequently degenerates due to sycophancy or cascading errors among correlated agents.
- **Correlated errors** cap information gain: an ensemble of $k$ agents with average pairwise error correlation $\bar{\phi}$ yields only $N_{\text{eff}} = \frac{k}{1 + (k-1)\bar{\phi}}$ independent votes.

### The Specific Novelty & Gap
While confidence-weighted voting and atomic claim reassembly exist independently:
**No existing work evaluates an external typed decision model (TypeSafe Jev `typesafe/jev-1.13` via Decisions API) as an atomic per-claim verifier across a heterogeneous multi-agent pool, coupled with held-out recalibration, agent error-correlation discounting, and a rigorous cost-accuracy frontier.**

Recent evaluations show Jev achieves competitive accuracy and superior Brier scores (e.g. 0.043 vs 0.054 against frontier LLM judges) with non-collapsed probability distributions across the $[0.01, 0.99]$ grid. However, Jev is vulnerable to:
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
- **Protocol:** Noul query repeated 3x. Compute raw vs recalibrated ECE, Brier, AUROC.
- **Robustness checks:** Option-name swap invariance (Yes/No vs True/False vs A/B), distractor perturbation.
- **Gate 1 Criterion:** Post-calibration ECE $\le 0.08$ and AUROC $\ge 0.75$. If failed, pivot Jev to selective routing/abstention only.

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
