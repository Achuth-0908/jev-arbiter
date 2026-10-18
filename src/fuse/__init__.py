"""Fusion and arbitration package."""

from .effective_votes import compute_pairwise_error_correlations, calculate_discounted_agreement
from .reassembly import reassemble_answer_from_claims
from .engine import ArbitratedClaim, ArbitrationResult, ArbitrationEngine

__all__ = [
    "compute_pairwise_error_correlations",
    "calculate_discounted_agreement",
    "reassemble_answer_from_claims",
    "ArbitratedClaim",
    "ArbitrationResult",
    "ArbitrationEngine",
]
