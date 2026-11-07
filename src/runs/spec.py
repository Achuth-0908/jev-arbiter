"""Declarative, hashable run specification. Every result traces back to one of these."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from ..config import canonical_hash, load_models_config


class DataSpec(BaseModel):
    max_per_subdataset_dev: int = 200
    max_per_subdataset_test: int = 200
    subsets: Optional[List[str]] = None


class RetrievalSpec(BaseModel):
    words_per_chunk: int = 120
    overlap: int = 20
    top_k: int = 2


class JevSpec(BaseModel):
    endpoint: str = "https://openrouter.ai/api/alpha/decisions"
    model: str = "typesafe/jev-1.13"
    repeats: int = 1
    derive_from_repeats: Optional[int] = None  # answer repeats=1 from cached repeats=N entries (no new calls)
    clip_bounds: Tuple[float, float] = (0.01, 0.99)
    concurrency: int = 8                       # max in-flight HTTP requests (and claims in flight)
    requests_per_second: Optional[float] = None  # optional global rate cap
    timeout_seconds: float = 60.0
    max_retries: int = 4
    backoff_factor: float = 1.5


class RunSpec(BaseModel):
    name: str = "phase1"
    mode: Literal["live", "mock"] = "mock"     # default is the safe, free one
    data: DataSpec = Field(default_factory=DataSpec)
    retrieval: RetrievalSpec = Field(default_factory=RetrievalSpec)
    jev: JevSpec = Field(default_factory=JevSpec)
    include_test: bool = False
    budget_usd: Optional[float] = None         # hard spend cap for live calls; aborts cleanly when reached
    output_dir: str = "results/runs"

    @classmethod
    def from_yaml(cls, path: Path) -> "RunSpec":
        """Load a run spec; Jev settings not given in the run file come from configs/models.yaml."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
        jev_defaults = {k: v for k, v in load_models_config().get("jev", {}).items()
                        if k in JevSpec.model_fields}
        # models.yaml uses concurrency_limit; the spec calls it concurrency
        if "concurrency_limit" in load_models_config().get("jev", {}):
            jev_defaults["concurrency"] = load_models_config()["jev"]["concurrency_limit"]
        raw["jev"] = {**jev_defaults, **(raw.get("jev") or {})}
        return cls(**raw)

    def spec_hash(self) -> str:
        # `include_test` and `output_dir` do not change what is measured on cal/val.
        d = self.model_dump(exclude={"include_test", "output_dir", "budget_usd"})
        return canonical_hash(d)
