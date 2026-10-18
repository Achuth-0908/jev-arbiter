"""Semantic clustering and normalization of atomic claims across candidate agents."""

from __future__ import annotations

import re
from typing import Dict, List, Set
from pydantic import BaseModel, Field

from .extractor import AtomicClaim


class ClaimCluster(BaseModel):
    """Cluster of equivalent atomic claims across candidate agents."""
    canonical_claim: str
    supporting_agents: List[str]
    agent_confidences: Dict[str, float]
    agreement_count: int
    agreement_ratio: float
    effective_agreement_ratio: float = 0.0
    mean_self_confidence: float
    claims: List[AtomicClaim] = Field(default_factory=list)


def stem_token(w: str) -> str:
    """Lightweight token stemming for robust clustering."""
    return re.sub(r"(ing|ed|es|s|tion|or|er)$", "", w.lower())


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity with stemming between two texts."""
    stopwords = {"that", "with", "from", "this", "were", "been", "have", "they", "will", "would", "about", "and", "the"}
    
    # Check if primary subjects conflict (e.g. Oppenheimer vs Lawrence)
    words_a = [w for w in re.findall(r"\b\w{3,}\b", text_a.lower()) if w not in stopwords]
    words_b = [w for w in re.findall(r"\b\w{3,}\b", text_b.lower()) if w not in stopwords]
    
    if not words_a or not words_b:
        return 0.0

    stems_a = set(stem_token(w) for w in words_a)
    stems_b = set(stem_token(w) for w in words_b)

    # If first subject word contradicts, lower similarity
    if words_a[0] != words_b[0] and stems_a != stems_b:
        first_a_in_b = stem_token(words_a[0]) in stems_b
        first_b_in_a = stem_token(words_b[0]) in stems_a
        if not first_a_in_b and not first_b_in_a:
            # Different subjects
            return len(stems_a.intersection(stems_b)) / (len(stems_a.union(stems_b)) + 2)

    return len(stems_a.intersection(stems_b)) / len(stems_a.union(stems_b))


def cluster_claims(
    claims_by_agent: Dict[str, List[AtomicClaim]],
    total_agents: int,
    similarity_threshold: float = 0.48,
) -> List[ClaimCluster]:
    """Group equivalent claims across different agents into unified clusters."""
    all_claims: List[AtomicClaim] = []
    for agent_id, agent_claims in claims_by_agent.items():
        all_claims.extend(agent_claims)

    clusters: List[List[AtomicClaim]] = []

    for claim in all_claims:
        matched_cluster = None
        best_sim = 0.0

        for cluster in clusters:
            sim = compute_token_jaccard(claim.claim_text, cluster[0].claim_text)
            if sim >= similarity_threshold and sim > best_sim:
                best_sim = sim
                matched_cluster = cluster

        if matched_cluster is not None:
            matched_cluster.append(claim)
        else:
            clusters.append([claim])

    result_clusters: List[ClaimCluster] = []
    for group in clusters:
        # Choose the longest, most complete claim as canonical
        canonical = max(group, key=lambda c: len(c.claim_text)).claim_text
        agents_set: Set[str] = set()
        conf_map: Dict[str, float] = {}

        for c in group:
            agents_set.add(c.source_agent_id)
            if c.source_agent_id not in conf_map or c.self_confidence > conf_map[c.source_agent_id]:
                conf_map[c.source_agent_id] = c.self_confidence

        supp_agents = sorted(list(agents_set))
        agreement_count = len(supp_agents)
        agreement_ratio = agreement_count / max(1, total_agents)
        mean_conf = float(sum(conf_map.values()) / max(1, len(conf_map)))

        result_clusters.append(
            ClaimCluster(
                canonical_claim=canonical,
                supporting_agents=supp_agents,
                agent_confidences=conf_map,
                agreement_count=agreement_count,
                agreement_ratio=agreement_ratio,
                mean_self_confidence=mean_conf,
                claims=group,
            )
        )

    # Sort descending by agreement count then mean confidence
    result_clusters.sort(key=lambda cl: (cl.agreement_count, cl.mean_self_confidence), reverse=True)
    return result_clusters
