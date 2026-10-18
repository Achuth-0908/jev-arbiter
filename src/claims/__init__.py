"""Claims processing package."""

from .extractor import AtomicClaim, ClaimExtractor
from .cluster import ClaimCluster, cluster_claims, compute_token_jaccard

__all__ = [
    "AtomicClaim",
    "ClaimExtractor",
    "ClaimCluster",
    "cluster_claims",
    "compute_token_jaccard",
]
