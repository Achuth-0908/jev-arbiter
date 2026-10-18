# Jev-Arbiter: Calibration-Weighted Multi-Agent Arbitration

This repository implements **claim-level, calibration-weighted multi-agent arbitration** using the TypeSafe Jev (`typesafe/jev-1.13`) Decisions API on OpenRouter.

Instead of naive majority-voting or self-consistency loops that fall prey to sycophancy and correlated errors, `jev-arbiter` breaks candidate reasoning into decontextualized atomic claims, retrieves concise BM25 grounding snippets, and queries Jev Noul verification with empirical recalibration and effective independent vote discounting:
$$N_{\text{eff}} = \frac{k}{1 + (k - 1) \bar{\phi}}$$

---

## Quickstart

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and configure your API keys (optional; offline mock mode runs out of the box):
```bash
cp .env.example .env
```
Key variables:
- `OPENROUTER_API_KEY`: OpenRouter API key for live calls to `typesafe/jev-1.13`.
- `DECISIONS_API_URL`: Points to `https://openrouter.ai/api/alpha/decisions`.
- `MOCK_API_MODE`: Set to `true` (default if no key) to run fully offline at $0 cost.

### 3. Run the Streamlit Interactive Platform
Launch the interactive dashboard to inspect arbitration live:
```bash
streamlit run app.py
```

### 4. Run the Test Suite
All tests run offline using the content-addressed cache and mock Jev simulator:
```bash
pytest tests/ -v
```

---

## Architecture Overview

```
[Candidate Agents: Claude, Llama, Gemini, Mistral]
                       │
                       ▼
         [Atomic Claim Extraction & Sanitizer]
                       │
                       ▼
          [Semantic Claim Clustering]
                       │
                       ▼
            [BM25 Snippet Retrieval]
                       │
                       ▼
      [Jev Decisions API (Noul, 3x repeats)]
                       │
                       ▼
         [Held-Out Recalibration Layer]
                       │
                       ▼
    [Calibrated Fusion: Jev + N_eff + Conf + BM25]
                       │
                       ▼
   [Decision Filter: Accepted vs Abstained vs Rejected]
                       │
                       ▼
         [Verified Final Answer Reassembly]
```

---

## Project Structure

```
jev-arbiter/
├── PLAN.md                  # Comprehensive master research & execution plan
├── AGENTS.md                # Agent rules, invariants, and key safety
├── app.py                   # Streamlit interactive testing platform
├── run_cli.py               # Fast CLI runner for probes and benchmarks
├── pyproject.toml           # Package configuration & dependencies
├── configs/
│   ├── models.yaml          # Jev endpoint, candidate models, parameters
│   ├── seeds.yaml           # Deterministic seeds for splits and bootstrap
│   └── thresholds.yaml      # Pre-registered acceptance thresholds and gates
├── src/
│   ├── clients/             # Jev Decisions client, LLM client, SQLite cache, mock
│   ├── data/                # LLM-AggreFact & HotpotQA loaders with frozen splits
│   ├── claims/              # Atomic claim extraction & semantic clustering
│   ├── retrieval/           # BM25 evidence indexing & retrieval
│   ├── verify/              # Jev Noul verification, repeats, option invariance
│   ├── calibrate/           # ECE, Brier, AUROC, Temperature & Isotonic scaling
│   ├── fuse/                # Pairwise correlation phi, N_eff, arbitration engine
│   └── eval/                # Cluster-bootstrap CIs, benchmark comparison runner
└── tests/                   # 100% offline-passing unit and integration tests
```

---

## Pre-Registration & Gate Criteria

- **Gate 1 (Phase 1):** Post-recalibration ECE $\le 0.08$ and AUROC $\ge 0.75$ on held-out LLM-AggreFact test split.
- **Gate 2 (Phase 3):** Statistically significant improvement ($p < 0.05$ via question-level cluster bootstrap) over unweighted majority voting and self-confidence weighting.
