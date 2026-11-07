"""Client for OpenRouter / TypeSafe Jev Decisions API with caching, retry, and mock fallback."""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from dotenv import load_dotenv

from .base import DecisionRequest, DecisionResponse, SingleDecisionResult
from .cache import CacheStore
from .mock_jev import MockJevSimulator

load_dotenv()

RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504)


class DecisionSchemaError(RuntimeError):
    """The Decisions API returned a payload we cannot interpret. Never papered over with a default."""


class FatalAPIError(RuntimeError):
    """Account/auth failure (401/402/403). Retrying cannot help; the run must stop loudly."""


class BudgetExceededError(FatalAPIError):
    """The configured spend cap would be exceeded by the next live call. Completed work stays cached."""


_FATAL_HINTS = {
    401: "invalid API key",
    402: "insufficient credits or key spending limit reached",
    403: "key not authorized for this endpoint",
}


def parse_live_answer(raw: Any) -> float:
    """Extract P(supported) from a live response, or raise DecisionSchemaError."""
    try:
        val = raw["answers"]["is_supported"]["noul"]
    except (KeyError, TypeError):
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise DecisionSchemaError(f"missing answers.is_supported.noul (top-level keys: {keys})") from None
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
        raise DecisionSchemaError(f"answers.is_supported.noul is not a probability: {type(val).__name__}")
    return float(val)


class JevDecisionsClient:
    """Client for TypeSafe Jev on the OpenRouter Decisions API.

    One pooled HTTP connection set per event loop, a global in-flight cap, an optional request
    rate limit, retries only for transient failures, and no silent fallbacks on bad payloads.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://openrouter.ai/api/alpha/decisions",
        model: str = "typesafe/jev-1.13",
        cache_store: Optional[CacheStore] = None,
        concurrency_limit: int = 5,
        timeout_seconds: float = 60.0,
        max_retries: int = 4,
        backoff_factor: float = 1.5,
        mock_mode: Optional[bool] = None,
        requests_per_second: Optional[float] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        max_spend_usd: Optional[float] = None,
        derive_from_repeats: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint or os.getenv("DECISIONS_API_URL", "https://openrouter.ai/api/alpha/decisions")
        self.model = model or os.getenv("DECISIONS_MODEL", "typesafe/jev-1.13")
        self.cache = cache_store or CacheStore()
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.requests_per_second = requests_per_second
        self._transport = transport
        # Spend cap (live calls only). Checked before each claim, so in-flight concurrent claims can
        # overshoot by at most concurrency x per-claim cost (a few hundredths of a cent).
        self.max_spend_usd = max_spend_usd
        self.spent_usd = 0.0
        self._cost_sum = 0.0
        self._cost_calls = 0
        # If set, a miss at repeats=r < N is answered from a cached repeats=N entry's first r repeats.
        self.derive_from_repeats = derive_from_repeats
        self.derived_from_cache = 0

        env_mock = os.getenv("MOCK_API_MODE", "false").lower() in ("true", "1", "yes")
        self.mock_mode = mock_mode if mock_mode is not None else (env_mock or not bool(self.api_key))
        self.simulator = MockJevSimulator()

        # Event-loop-bound resources, created lazily (recreated if the loop changes).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._sem: Optional[asyncio.Semaphore] = None
        self._rl_lock: Optional[asyncio.Lock] = None
        self._next_slot = 0.0
        self.network_calls = 0  # live HTTP requests actually sent (for audits / idempotency checks)

    def _ensure_loop_resources(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._http is None or self._http.is_closed:
            self._loop = loop
            n = max(1, self.concurrency_limit)
            self._http = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=n, max_keepalive_connections=n),
                transport=self._transport,
            )
            self._sem = asyncio.Semaphore(n)
            self._rl_lock = asyncio.Lock()
            self._next_slot = 0.0
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()

    async def _throttle(self) -> None:
        if not self.requests_per_second:
            return
        assert self._rl_lock is not None
        async with self._rl_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + 1.0 / self.requests_per_second
        if wait > 0:
            await asyncio.sleep(wait)

    def _backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        if retry_after is not None:
            try:
                return max(0.0, min(60.0, float(retry_after)))
            except ValueError:
                pass
        return (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)

    async def _call_single_decision_live(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One HTTP POST. Retries transient failures only; permanent 4xx errors raise immediately."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jev-arbiter/research",
            "X-Title": "Jev Multi-Agent Arbiter",
        }
        assert self._sem is not None
        last_err = "no attempt made"
        for attempt in range(self.max_retries):
            await self._throttle()
            try:
                async with self._sem:
                    self.network_calls += 1
                    resp = await client.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            except httpx.RequestError as e:
                last_err = type(e).__name__
                await asyncio.sleep(self._backoff(attempt))
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    raise DecisionSchemaError("HTTP 200 with a non-JSON body") from None
            if resp.status_code in RETRYABLE_STATUS:
                last_err = f"HTTP {resp.status_code}"
                await asyncio.sleep(self._backoff(attempt, resp.headers.get("retry-after")))
                continue
            if resp.status_code in _FATAL_HINTS:
                raise FatalAPIError(
                    f"Decisions API returned non-retryable HTTP {resp.status_code} ({_FATAL_HINTS[resp.status_code]})"
                )
            raise RuntimeError(f"Decisions API returned non-retryable HTTP {resp.status_code}")
        raise RuntimeError(f"Decisions API request failed after {self.max_retries} attempts: {last_err}")

    async def evaluate_claim(
        self,
        claim: str,
        evidence: str,
        options: Optional[List[str]] = None,
        criteria: str = "Is the claim strictly supported by the provided evidence?",
        repeats: int = 3,
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
        cache_salt: str = "",
    ) -> DecisionResponse:
        """Evaluate a claim against evidence with n repeats, caching, and variance tracking."""
        options = options or ["Yes", "No"]
        state = {"claim": claim, "evidence": evidence}

        req = DecisionRequest(
            model=self.model,
            decision_type="noul",
            state=state,
            options=options,
            criteria=criteria,
            repeats=repeats,
            cache_salt=cache_salt,
        )
        cache_key = req.cache_key(self.endpoint, mode="mock" if self.mock_mode else "live")

        mode = "mock" if self.mock_mode else "live"
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            data, _lat, _cost = cached_entry
            resp = DecisionResponse(**data)
            resp.cached = True
            return resp

        derived = self._derive_from_cache(req, repeats, options, clip_bounds, mode)
        if derived is not None:
            self.cache.set(key=cache_key, endpoint=self.endpoint, model=self.model,
                           response_payload=derived.model_dump(), latency_ms=derived.latency_ms,
                           cost_usd=derived.cost_usd)
            self.derived_from_cache += 1
            return derived

        if not self.mock_mode and self.max_spend_usd is not None:
            # Estimate from observed per-call cost only; the first call is never blocked by a guess.
            est = (self._cost_sum / self._cost_calls) if self._cost_calls else 0.0
            if self.spent_usd + est * repeats > self.max_spend_usd:
                raise BudgetExceededError(
                    f"Spend cap ${self.max_spend_usd:.4f} reached (spent ${self.spent_usd:.4f}); "
                    "completed work is cached - raise the budget to resume"
                )

        start_time = time.time()
        results: List[SingleDecisionResult] = []
        total_cost = 0.0
        pos_opt = options[0]
        neg_opt = options[1] if len(options) > 1 else "No"
        resolved_version = "mock-jev"

        if self.mock_mode:
            for idx in range(repeats):
                raw = self.simulator.evaluate(state=state, options=options, criteria=criteria, run_idx=idx)
                choice = raw["choices"][0]
                results.append(
                    SingleDecisionResult(
                        chosen_option=choice["decision"],
                        probabilities=choice["probabilities"],
                        raw_response=raw,
                    )
                )
                total_cost += raw.get("usage", {}).get("cost_usd", 0.00025)
        else:
            http = self._ensure_loop_resources()
            # Note: the live API takes no `options`; they only name the keys of our result dict.
            questions_payload = {
                "is_supported": {
                    "type": "noul",
                    "instructions": criteria,
                    "criteria": {
                        "true": f"The claim '{claim}' is directly and strictly supported by the evidence.",
                        "false": f"The claim '{claim}' is not supported or is contradicted by the evidence.",
                    },
                }
            }
            payload = {"model": self.model, "state": state, "questions": questions_payload}
            raw_responses = await asyncio.gather(
                *[self._call_single_decision_live(http, payload) for _ in range(repeats)]
            )
            for raw in raw_responses:
                p_true = parse_live_answer(raw)
                results.append(
                    SingleDecisionResult(
                        chosen_option=pos_opt if p_true >= 0.50 else neg_opt,
                        probabilities={pos_opt: p_true, neg_opt: round(1.0 - p_true, 4)},
                        raw_response=raw,
                    )
                )
                usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                total_cost += float(usage.get("cost", usage.get("cost_usd", 0.0)))
            first = raw_responses[0]
            if isinstance(first, dict) and isinstance(first.get("model"), str):
                resolved_version = first["model"]

        latency_ms = (time.time() - start_time) * 1000.0
        response = self._aggregate(results, options, clip_bounds, latency_ms, total_cost, resolved_version, False)
        if not self.mock_mode:
            self.spent_usd += total_cost
            self._cost_sum += total_cost
            self._cost_calls += repeats

        self.cache.set(
            key=cache_key,
            endpoint=self.endpoint,
            model=self.model,
            response_payload=response.model_dump(),
            latency_ms=latency_ms,
            cost_usd=total_cost,
        )
        return response

    def _aggregate(
        self,
        results: List[SingleDecisionResult],
        options: List[str],
        clip_bounds: Tuple[float, float],
        latency_ms: float,
        cost_usd: float,
        resolved_version: str,
        cached: bool,
    ) -> DecisionResponse:
        pos_opt = options[0]
        mean_probs: Dict[str, float] = {}
        for opt in options:
            vals = [res.probabilities.get(opt, 0.0) for res in results]
            mean_probs[opt] = float(sum(vals) / len(vals))
        pos_vals = [res.probabilities.get(pos_opt, 0.0) for res in results]
        var = (
            float(sum((x - mean_probs[pos_opt]) ** 2 for x in pos_vals) / len(pos_vals))
            if len(pos_vals) > 1
            else 0.0
        )
        low_b, high_b = clip_bounds
        clipped = {opt: max(low_b, min(high_b, round(prob, 4))) for opt, prob in mean_probs.items()}
        return DecisionResponse(
            model=self.model, resolved_version=resolved_version, results=results,
            mean_probabilities=mean_probs, clipped_probabilities=clipped, variance=var,
            latency_ms=latency_ms, cost_usd=cost_usd, cached=cached,
        )

    def _derive_from_cache(
        self, req: DecisionRequest, repeats: int, options: List[str],
        clip_bounds: Tuple[float, float], mode: str,
    ) -> Optional[DecisionResponse]:
        """First `repeats` stored repeats of a larger cached entry for the identical request."""
        n = self.derive_from_repeats
        if not n or repeats >= n:
            return None
        src = self.cache.get(req.model_copy(update={"repeats": n}).cache_key(self.endpoint, mode=mode))
        if not src:
            return None
        full = DecisionResponse(**src[0])
        if len(full.results) < repeats:
            return None
        used = full.results[:repeats]
        cost = 0.0
        for r in used:
            usage = r.raw_response.get("usage", {}) if isinstance(r.raw_response, dict) else {}
            cost += float(usage.get("cost", usage.get("cost_usd", 0.0)))
        # cached=True: no new external call was made to produce this view.
        return self._aggregate(used, options, clip_bounds, full.latency_ms, cost, full.resolved_version, True)
