"""Base data models, hashing, and abstract client definitions."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def canonical_json_hash(payload: Any) -> str:
    """Generate deterministic SHA256 hash for arbitrary nested dicts/lists."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class DecisionRequest(BaseModel):
    """Request model for Jev Decisions API."""
    model: str = "typesafe/jev-1.13"
    decision_type: str = "noul"
    state: Dict[str, Any] = Field(default_factory=dict)
    options: List[str] = Field(default_factory=lambda: ["Yes", "No"])
    criteria: str = "Is the claim strictly supported by the provided evidence?"
    repeats: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def cache_key(self, endpoint: str) -> str:
        data = {
            "endpoint": endpoint,
            "model": self.model,
            "decision_type": self.decision_type,
            "state": self.state,
            "options": self.options,
            "criteria": self.criteria,
        }
        return canonical_json_hash(data)


class SingleDecisionResult(BaseModel):
    """Result of a single decision iteration."""
    chosen_option: str
    probabilities: Dict[str, float]
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    """Aggregated response model for Jev Decisions."""
    model: str
    resolved_version: str = "jev-1.13-default"
    results: List[SingleDecisionResult]
    mean_probabilities: Dict[str, float]
    clipped_probabilities: Dict[str, float]
    variance: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    timestamp: float = Field(default_factory=time.time)


class LLMRequest(BaseModel):
    """Standard request model for text completions/chat."""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 512
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def cache_key(self, endpoint: str) -> str:
        data = {
            "endpoint": endpoint,
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return canonical_json_hash(data)


class LLMResponse(BaseModel):
    """Standard response model for text completions/chat."""
    model: str
    content: str
    self_confidence: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    timestamp: float = Field(default_factory=time.time)
