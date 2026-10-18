"""Multi-agent arbitration engine with calibration weighting, effective votes, and fusion."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from scipy.special import expit, logit

from ..claims.extractor import AtomicClaim, ClaimExtractor
from ..claims.cluster import ClaimCluster, cluster_claims
from ..retrieval.bm25 import BM25SnippetRetriever
from ..verify.verifier import JevClaimVerifier, ClaimVerificationResult
from .effective_votes import calculate_discounted_agreement
from .reassembly import reassemble_answer_from_claims


class ArbitratedClaim(BaseModel):
    """Result of arbitrating an individual claim cluster."""
    canonical_claim: str
    supporting_agents: List[str]
    agreement_count: int
    raw_agreement_ratio: float
    effective_agreement_ratio: float
    mean_self_confidence: float
    bm25_score: float
    grounding_snippet: str
    raw_jev_prob: float
    calibrated_jev_prob: float
    fused_prob: float
    status: str  # ACCEPTED, REJECTED, ABSTAINED
    variance_across_repeats: float = 0.0


class ArbitrationResult(BaseModel):
    """Aggregate result of multi-agent arbitration for a query."""
    strategy: str
    arbitrated_claims: List[ArbitratedClaim]
    accepted_claims: List[str]
    rejected_claims: List[str]
    abstained_claims: List[str]
    final_answer: str
    mean_error_phi: float
    effective_independent_votes: float
    total_cost_usd: float
    total_latency_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArbitrationEngine:
    """Orchestrates end-to-end multi-agent arbitration."""

    def __init__(
        self,
        verifier: Optional[JevClaimVerifier] = None,
        extractor: Optional[ClaimExtractor] = None,
        acceptance_threshold: float = 0.65,
        abstention_threshold: float = 0.35,
        default_mean_phi: float = 0.30,
    ):
        self.verifier = verifier or JevClaimVerifier()
        self.extractor = extractor or ClaimExtractor()
        self.acceptance_threshold = acceptance_threshold
        self.abstention_threshold = abstention_threshold
        self.mean_phi = default_mean_phi

    def compute_fused_probability(
        self,
        calibrated_jev_prob: float,
        effective_agreement: float,
        mean_self_conf: float,
        bm25_score: float,
    ) -> float:
        """Combine calibrated Jev probability, correlation-discounted agreement, confidence, and retrieval score.
        
        Using calibrated log-odds fusion with empirical weights:
        w_jev = 0.65, w_agree = 0.20, w_self = 0.10, w_bm25 = 0.05
        """
        p_jev = float(np.clip(calibrated_jev_prob, 0.01, 0.99))
        l_jev = float(logit(p_jev))

        p_agree = float(np.clip(effective_agreement, 0.01, 0.99))
        l_agree = float(logit(p_agree))

        p_self = float(np.clip(mean_self_conf, 0.01, 0.99))
        l_self = float(logit(p_self))

        bm25_norm = float(np.clip(max(0.0, bm25_score) / 10.0, 0.0, 1.0))
        p_bm25 = float(np.clip(0.15 + 0.70 * bm25_norm, 0.01, 0.99))
        l_bm25 = float(logit(p_bm25))

        fused_logit = (0.65 * l_jev) + (0.20 * l_agree) + (0.10 * l_self) + (0.05 * l_bm25)
        return float(expit(fused_logit))

    async def arbitrate(
        self,
        query: str,
        candidate_responses: Dict[str, str],
        candidate_confidences: Optional[Dict[str, float]] = None,
        reference_documents: Optional[List[str]] = None,
        strategy: str = "calibrated_fusion",
        mean_phi: Optional[float] = None,
    ) -> ArbitrationResult:
        """Run full arbitration across candidate agents."""
        start_time = time.time()
        confidences = candidate_confidences or {k: 0.80 for k in candidate_responses}
        phi = self.mean_phi if mean_phi is None else mean_phi
        total_agents = len(candidate_responses)

        # 1. Extract atomic claims per candidate
        claims_by_agent: Dict[str, List[AtomicClaim]] = {}
        for agent_id, text in candidate_responses.items():
            agent_conf = confidences.get(agent_id, 0.80)
            claims = await self.extractor.extract(text, agent_id, agent_conf)
            claims_by_agent[agent_id] = claims

        # 2. Semantic clustering across agents
        clusters = cluster_claims(claims_by_agent, total_agents=total_agents)

        # 3. Setup BM25 Retriever
        doc_pool = list(reference_documents or [])
        if not doc_pool:
            doc_pool = list(candidate_responses.values())
        retriever = BM25SnippetRetriever(doc_pool)

        # 4. Verify each cluster
        arbitrated_list: List[ArbitratedClaim] = []
        total_cost = 0.0

        for cl in clusters:
            # Retrieve evidence snippet
            evidence_snippet, bm25_sc = retriever.get_grounding_context(cl.canonical_claim, top_k=2)

            # Jev Verification
            v_res = await self.verifier.verify(
                claim=cl.canonical_claim,
                evidence=evidence_snippet,
            )
            total_cost += v_res.cost_usd

            eff_agreement = calculate_discounted_agreement(
                votes_for=cl.agreement_count,
                total_agents=total_agents,
                mean_phi=phi,
            )

            fused_p = self.compute_fused_probability(
                calibrated_jev_prob=v_res.calibrated_p_supported,
                effective_agreement=eff_agreement,
                mean_self_conf=cl.mean_self_confidence,
                bm25_score=bm25_sc,
            )

            # Select score based on strategy
            if strategy == "majority_vote":
                score = cl.agreement_ratio
            elif strategy == "self_confidence":
                score = cl.agreement_ratio * cl.mean_self_confidence
            elif strategy == "raw_jev":
                score = v_res.raw_p_supported
            elif strategy == "calibrated_jev":
                score = v_res.calibrated_p_supported
            else:  # calibrated_fusion
                score = fused_p

            if score >= self.acceptance_threshold:
                status = "ACCEPTED"
            elif score < self.abstention_threshold:
                status = "REJECTED"
            else:
                status = "ABSTAINED"

            arbitrated_list.append(
                ArbitratedClaim(
                    canonical_claim=cl.canonical_claim,
                    supporting_agents=cl.supporting_agents,
                    agreement_count=cl.agreement_count,
                    raw_agreement_ratio=cl.agreement_ratio,
                    effective_agreement_ratio=eff_agreement,
                    mean_self_confidence=cl.mean_self_confidence,
                    bm25_score=bm25_sc,
                    grounding_snippet=evidence_snippet,
                    raw_jev_prob=v_res.raw_p_supported,
                    calibrated_jev_prob=v_res.calibrated_p_supported,
                    fused_prob=fused_p,
                    status=status,
                    variance_across_repeats=v_res.variance_across_repeats,
                )
            )

        accepted = [ac.canonical_claim for ac in arbitrated_list if ac.status == "ACCEPTED"]
        rejected = [ac.canonical_claim for ac in arbitrated_list if ac.status == "REJECTED"]
        abstained = [ac.canonical_claim for ac in arbitrated_list if ac.status == "ABSTAINED"]

        # 5. Reassemble Answer
        final_answer = reassemble_answer_from_claims(accepted, query=query)

        latency_ms = (time.time() - start_time) * 1000.0
        n_eff = total_agents / (1.0 + (total_agents - 1) * max(0.0, min(1.0, phi)))

        return ArbitrationResult(
            strategy=strategy,
            arbitrated_claims=arbitrated_list,
            accepted_claims=accepted,
            rejected_claims=rejected,
            abstained_claims=abstained,
            final_answer=final_answer,
            mean_error_phi=phi,
            effective_independent_votes=n_eff,
            total_cost_usd=total_cost,
            total_latency_ms=latency_ms,
        )
