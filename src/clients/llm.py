"""Unified LLM client for candidates, extraction, and baseline judges with persistent caching."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

from .base import LLMRequest, LLMResponse
from .cache import CacheStore

load_dotenv()


class UnifiedLLMClient:
    """Async LLM client supporting OpenRouter with robust caching and mock capability."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        cache_store: Optional[CacheStore] = None,
        timeout_seconds: float = 45.0,
        mock_mode: Optional[bool] = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.endpoint = endpoint
        self.cache = cache_store or CacheStore()
        self.timeout = timeout_seconds

        env_mock = os.getenv("MOCK_API_MODE", "false").lower() in ("true", "1", "yes")
        self.mock_mode = mock_mode if mock_mode is not None else (env_mock or not bool(self.api_key))

    def _extract_confidence(self, text: str) -> Optional[float]:
        """Parse verbalized self-reported confidence from text (e.g., 'Confidence: 0.85')."""
        match = re.search(r"confidence\s*[:=-]\s*([0-1]?\.\d+|\d+%)", text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            if raw.endswith("%"):
                try:
                    return float(raw.strip("%")) / 100.0
                except ValueError:
                    return None
            try:
                val = float(raw)
                return min(1.0, max(0.0, val))
            except ValueError:
                return None
        return None

    def _simulate_agent_generation(self, model: str, prompt: str) -> str:
        """Deterministic mock response for candidate agents and extractors."""
        if "extract" in prompt.lower() or "atomic claims" in prompt.lower():
            # Mock extraction
            lines = [l.strip() for l in prompt.split("\n") if len(l.strip()) > 10 and not l.startswith("Extract")]
            sample_claim = lines[-1] if lines else "The system operates as specified."
            return (
                f"- Claim 1: {sample_claim[:80]}\n"
                f"- Claim 2: The verification signal provides calibrated boundaries."
            )

        if "judge" in prompt.lower() or "verify" in prompt.lower():
            return "VERDICT: Supported\nConfidence: 0.82\nExplanation: The claim is grounded in the source text."

        return (
            f"Based on the provided context, the primary factual outcome is supported.\n"
            f"Confidence: 0.85"
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Call LLM completion with content-addressable caching."""
        req = LLMRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cache_key = req.cache_key(self.endpoint, mode="mock" if self.mock_mode else "live")

        # Check Cache
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            data, _, _ = cached_entry
            resp = LLMResponse(**data)
            resp.cached = True
            return resp

        start_time = time.time()

        if self.mock_mode:
            await asyncio.sleep(0.01)
            last_prompt = messages[-1].get("content", "") if messages else ""
            content = self._simulate_agent_generation(model, last_prompt)
            prompt_tokens = len(last_prompt.split()) + 20
            comp_tokens = len(content.split())
            cost = (prompt_tokens * 0.0000002) + (comp_tokens * 0.0000008)
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            comp_tokens = usage.get("completion_tokens", 0)
            cost = usage.get("cost_usd", 0.0005)

        latency_ms = (time.time() - start_time) * 1000.0
        conf = self._extract_confidence(content)

        response = LLMResponse(
            model=model,
            content=content,
            self_confidence=conf if conf is not None else 0.75,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=prompt_tokens + comp_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            cached=False,
        )

        self.cache.set(
            key=cache_key,
            endpoint=self.endpoint,
            model=model,
            response_payload=response.model_dump(),
            latency_ms=latency_ms,
            cost_usd=cost,
        )

        return response
