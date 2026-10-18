"""Data package."""

from .loaders import (
    FactClaimInstance,
    generate_benchmark_synthetic_subset,
    create_frozen_splits,
)
from .qa_loader import MultiHopQAInstance, get_sample_multihop_instances

__all__ = [
    "FactClaimInstance",
    "generate_benchmark_synthetic_subset",
    "create_frozen_splits",
    "MultiHopQAInstance",
    "get_sample_multihop_instances",
]
