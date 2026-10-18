"""Client for OpenRouter / TypeSafe Jev Decisions API with caching, retry, and mock fallback."""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

from .base import DecisionRequest, DecisionResponse, SingleDecisionResult
from .cache import CacheStore
from .mock_jev import MockJevSimulator

load_dotenv()


class JevDecisionsClient:
    """Production-grade client for TypeSafe Jev on OpenRouter Decisions API."""

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
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint or os.getenv("DECISIONS_API_URL", "https://openrouter.ai/api/alpha/decisions")
        self.model = model or os.getenv("DECISIONS_MODEL", "typesafe/jev-1.13")
        self.cache = cache_store or CacheStore()
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Determine mock mode: explicit flag, env var, or missing API key
        env_mock = os.getenv("MOCK_API_MODE", "false").lower() in ("true", "1", "yes")
        self.mock_mode = mock_mode if mock_mode is not None else (env_mock or not bool(self.api_key))
        self.simulator = MockJevSimulator()

    async def _call_single_decision_live(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single HTTP POST request to OpenRouter Decisions endpoint with backoff."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jev-arbiter/research",
            "X-Title": "Jev Multi-Agent Arbiter",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    resp = await client.post(
                        self.endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (429, 500, 502, 503, 504):
                    wait_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(wait_time)
                else:
                    resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_err = e
                wait_time = (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"Decisions API request failed after {self.max_retries} retries: {last_err}")

    async def evaluate_claim(
        self,
        claim: str,
        evidence: str,
        options: Optional[List[str]] = None,
        criteria: str = "Is the claim strictly supported by the provided evidence?",
        repeats: int = 3,
        clip_bounds: Tuple[float, float] = (0.01, 0.99),
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
        )
        cache_key = req.cache_key(self.endpoint)

        # Check Cache first
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            data, lat_ms, cost = cached_entry
            resp = DecisionResponse(**data)
            resp.cached = True
            return resp

        # Cache Miss: Execute evaluation
        start_time = time.time()
        results: List[SingleDecisionResult] = []
        total_cost = 0.0

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
            async with httpx.AsyncClient() as http_client:
                tasks = []
                pos_opt = options[0]
                neg_opt = options[1] if len(options) > 1 else "No"
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
                for idx in range(repeats):
                    payload = {
                        "model": self.model,
                        "state": state,
                        "questions": questions_payload,
                    }
                    tasks.append(self._call_single_decision_live(http_client, payload))

                raw_responses = await asyncio.gather(*tasks)
                for raw in raw_responses:
                    # Check for live OpenRouter answers schema
                    if "answers" in raw and "is_supported" in raw["answers"]:
                        ans = raw["answers"]["is_supported"]
                        p_true = float(ans.get("noul", 0.50))
                        probs = {pos_opt: p_true, neg_opt: round(1.0 - p_true, 4)}
                        chosen = pos_opt if p_true >= 0.50 else neg_opt
                    elif "choices" in raw:
                        choice = raw["choices"][0]
                        chosen = choice["decision"]
                        probs = choice["probabilities"]
                    else:
                        chosen = pos_opt
                        probs = {pos_opt: 0.50, neg_opt: 0.50}

                    results.append(
                        SingleDecisionResult(
                            chosen_option=chosen,
                            probabilities=probs,
                            raw_response=raw,
                        )
                    )
                    cost_val = raw.get("usage", {}).get("cost", raw.get("usage", {}).get("cost_usd", 0.00002))
                    total_cost += float(cost_val)

        latency_ms = (time.time() - start_time) * 1000.0

        # Compute Mean Probabilities
        pos_option = options[0]
        mean_probs: Dict[str, float] = {}
        for opt in options:
            vals = [res.probabilities.get(opt, 0.0) for res in results]
            mean_probs[opt] = float(sum(vals) / len(vals))

        # Variance of primary positive option across repeated runs
        pos_vals = [res.probabilities.get(pos_option, 0.0) for res in results]
        var = float(sum((x - mean_probs[pos_option]) ** 2 for x in pos_vals) / len(pos_vals)) if len(pos_vals) > 1 else 0.0

        # Clipped Probabilities to [0.01, 0.99]
        low_b, high_b = clip_bounds
        clipped_probs = {
            opt: max(low_b, min(high_b, round(prob, 4))) for opt, prob in mean_probs.items()
        }

        response = DecisionResponse(
            model=self.model,
            resolved_version="jev-1.13",
            results=results,
            mean_probabilities=mean_probs,
            clipped_probabilities=clipped_probs,
            variance=var,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            cached=False,
        )

        # Store in cache
        self.cache.set(
            key=cache_key,
            endpoint=self.endpoint,
            model=self.model,
            response_payload=response.model_dump(),
            latency_ms=latency_ms,
            cost_usd=total_cost,
        )

        return response
