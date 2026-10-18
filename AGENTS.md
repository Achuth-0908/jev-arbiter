# Agent Rules & Invariants (`AGENTS.md`)

## 1. Security & API Key Invariants
- **NEVER** print, log, echo, or commit API keys (`OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, etc.).
- Always read environment variables from `.env` using `python-dotenv`.
- If an API key is missing or invalid, fall back gracefully to mock/offline replay mode rather than crashing or throwing key values.
- Never write credentials or Authorization headers to cache files or error logs.

## 2. Caching & Reproducibility Invariants
- **Every external call** (OpenRouter Decisions API, LLM completion, embedding, retrieval) must pass through `src/clients/cache.py`.
- The cache key MUST be deterministic and content-addressed: SHA256 of `(endpoint, model, json_canonical(state), json_canonical(questions))`.
- All runs must be idempotent: running the exact same experiment twice MUST produce identical results with 0 extra network calls.
- All pseudo-random processes (splits, bootstrap resampling, clustering, tie-breaking) MUST specify an explicit integer seed loaded from `configs/seeds.yaml`.

## 3. Scientific & Evaluation Protocol
- **Split Purity:** The test split MUST remain untouched during prompt tuning, hyperparameter search, and threshold calibration.
- **Pre-Registration:** Before running test set evaluations, write down and commit target thresholds and metric criteria into `configs/thresholds.yaml`.
- **Confidence Intervals:** Never report point estimates alone for key comparisons. Use cluster-bootstrap resampling grouped at the question/item level (to account for within-question claim correlation) with at least 1,000 bootstrap iterations.
- **Reporting Invariance:** Always log both raw and recalibrated ECE and Brier scores, alongside effective independent votes $N_{\text{eff}}$.

## 4. Testing & Code Quality
- Offline test suite must execute cleanly without internet access (`pytest tests/`).
- Code must use type hints and Pydantic models for structured payloads and configuration.
- Keep components modular: claims, retrieval, verification, calibration, and fusion must remain decoupled.
