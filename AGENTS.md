# Agent Rules & Invariants (`AGENTS.md`)

## 1. Security & API Key Invariants
- **NEVER** print, log, echo, or commit API keys (`OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, etc.).
- Always read environment variables from `.env` using `python-dotenv`.
- If an API key is missing or invalid, fall back gracefully to mock/offline replay mode rather than crashing or throwing key values.
- Never write credentials or Authorization headers to cache files or error logs.

## 2. Caching & Reproducibility Invariants
- **Every external call** (OpenRouter Decisions API, LLM completion, embedding, retrieval) must pass through `src/clients/cache.py`.
- The cache key MUST be deterministic and content-addressed: SHA256 of `(endpoint, mode, model, decision_type, json_canonical(state), options, criteria, repeats[, cache_salt])`. `mode` (`live`|`mock`) and `repeats` are part of the key so mock output can never be served as live and a 1-repeat result never as a 3-repeat one. `cache_salt` is added only when non-empty, to force an independent call (nondeterminism probes). *(Amended: the original spec omitted mode and repeats, which let mock results leak into live runs.)*
- A request for `repeats=r` MAY be answered from a cached entry for the identical request with `repeats=N > r` by taking its first `r` stored repeats (`derive_from_repeats`). This is a deterministic view of stored raw data, is counted as cached, and never crosses different criteria or salts.
- All runs must be idempotent: running the exact same experiment twice MUST produce identical results with 0 extra network calls.
- All pseudo-random processes (splits, bootstrap resampling, clustering, tie-breaking) MUST specify an explicit integer seed loaded from `configs/seeds.yaml`.
- Every run is described by a hashed `RunSpec` (`configs/runs/*.yaml`); the run ID is the hash of the spec, thresholds, seeds and split fingerprints, and each run writes a `manifest.json` (code commit, dirty flag, registration state, costs, error classes).

## 2b. Failure Handling Invariants
- **No silent fallbacks.** An unrecognised API payload raises `DecisionSchemaError`; it is never replaced by a default probability. LLM-extraction failures must be recorded, not hidden.
- **Auth/billing errors are fatal.** HTTP 401/402/403 raise `FatalAPIError` and abort the run (no retries, no per-claim error rows). Other 4xx errors are not retried; only transient statuses (408/425/429/5xx) are.
- **Live runs must be live.** A run whose spec says `mode: live` refuses to start if the client would fall back to mock.
- **Spend is capped.** Runs set `budget_usd`; the client raises `BudgetExceededError` before a call that would exceed it. Finished work stays cached, so a run resumes at no cost.
- Failures and fallbacks must appear in the run report (error classes, probe failure counts), not only in logs.

## 3. Scientific & Evaluation Protocol
- **Split Purity:** The test split MUST remain untouched during prompt tuning, hyperparameter search, and threshold calibration.
- **Pre-Registration:** Before running test set evaluations, write down and commit target thresholds and metric criteria into `configs/thresholds.yaml`. Enforced in code: `python -m src.runs register` writes `configs/thresholds.lock.json` (hash of the parsed thresholds), and the runner refuses `--include-test` unless the lock matches. The manifest records whether both files are committed and clean; a run whose registration is not committed must say so and cannot claim formal pre-registration. The run spec's `jev.repeats` must equal the registered `phase1.repeats`.
- **Test-split access:** the test split is scored only via `--include-test`, once, after registration. Probes, calibrator selection and any protocol choice (e.g. `repeats`) are made on calibration/validation data only.
- **Confidence Intervals:** Never report point estimates alone for key comparisons. Use cluster-bootstrap resampling grouped at the question/item level (to account for within-question claim correlation) with at least 1,000 bootstrap iterations.
- **Reporting Invariance:** Always log both raw and recalibrated ECE and Brier scores, alongside effective independent votes $N_{\text{eff}}$.

## 4. Testing & Code Quality
- Offline test suite must execute cleanly without internet access (`pytest tests/`).
- Code must use type hints and Pydantic models for structured payloads and configuration.
- Keep components modular: claims, retrieval, verification, calibration, and fusion must remain decoupled.

## 5. Deviations Log (be explicit; never hide)
- 2026-11-07: Gate 1 `target_brier_max: 0.12` kept as written although `PLAN.md`'s original Gate 1 listed only ECE and AUROC. Validation Brier (about 0.14) suggests this criterion will fail; the failure will be reported, not amended away.
- 2026-11-07: Phase 1 `repeats` changed 3 -> 1 on calibration/validation evidence alone (paired AUROC difference -0.0004, 0.9% decision flips, about one third of the cost), registered before the test split is scored.
- 2026-11-07: The option-name-swap invariance check was replaced by a wording-invariance probe: the live Decisions API takes no `options`, so swapping them sent an identical request and measured nothing.
