"""High-fidelity mock simulator for TypeSafe Jev Decisions API."""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any, Dict, List, Optional


class MockJevSimulator:
    """Simulates TypeSafe Jev Decisions API behavior offline with realistic properties.
    
    Properties simulated from literature:
    1. Grid rounding to 0.01.
    2. Usable middle band across [0.01, 0.99] without LLM judge extreme collapse.
    3. ~1.33% non-determinism across identical runs.
    4. Realistic cost ($0.00025 per noul evaluation).
    5. Sensitivity to option naming, ungrounded entities, and evidence overlap.
    """

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    @staticmethod
    def _stem(w: str) -> str:
        """Lightweight suffix stripping for token matching."""
        return re.sub(r"(ing|ed|es|s|tion)$", "", w.lower())

    def _estimate_semantic_support(self, claim: str, evidence: str) -> float:
        """Heuristic semantic overlap and consistency score between claim and evidence."""
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        # Check for explicit contradiction / negation markers
        negations = [" not ", " never ", " failed ", " incorrect ", " false ", " refused "]
        claim_has_neg = any(neg in claim_lower for neg in negations)
        ev_has_neg = any(neg in evidence_lower for neg in negations)
        contradiction = (claim_has_neg != ev_has_neg) if (claim_has_neg or ev_has_neg) else False

        # Extract content words (length >= 3, excluding stopwords)
        stopwords = {"that", "with", "from", "this", "were", "been", "have", "they", "will", "would", "about"}
        claim_raw_words = [
            w.strip(".,;:?!'\"()[]") for w in claim_lower.split()
            if len(w) >= 3 and w not in stopwords
        ]
        ev_raw_words = [
            w.strip(".,;:?!'\"()[]") for w in evidence_lower.split()
            if len(w) >= 3 and w not in stopwords
        ]

        if not claim_raw_words:
            return 0.50

        claim_stems = set(self._stem(w) for w in claim_raw_words)
        ev_stems = set(self._stem(w) for w in ev_raw_words)

        overlap = len(claim_stems.intersection(ev_stems)) / len(claim_stems)

        # Numerical check: dates/numbers in claim must be present in evidence
        claim_numbers = set(w for w in claim_raw_words if any(char.isdigit() for char in w))
        ev_numbers = set(w for w in ev_raw_words if any(char.isdigit() for char in w))
        if claim_numbers:
            num_overlap = len(claim_numbers.intersection(ev_numbers)) / len(claim_numbers)
            if num_overlap < 1.0:
                overlap *= 0.30

        # Named Entity check: proper nouns in claim must be present in evidence
        claim_entities = set(
            w.strip(".,;:?!'\"()[]") for w in claim.split()
            if w and w[0].isupper() and len(w) >= 3 and w.lower() not in stopwords
        )
        if claim_entities:
            unmatched_entities = [
                ent for ent in claim_entities
                if self._stem(ent) not in ev_stems
            ]
            if unmatched_entities:
                # Penalize ungrounded entity claims (e.g. Stanford when evidence only has Berkeley)
                overlap *= 0.45

        if contradiction or overlap < 0.60:
            # Low grounding, hallucinated entity, or contradiction
            base_p = 0.05 + (overlap * 0.35)
        else:
            # Substantial grounding without contradiction
            base_p = 0.65 + (overlap * 0.30)

        return min(0.98, max(0.02, base_p))

    def evaluate(
        self,
        state: Dict[str, Any],
        options: Optional[List[str]] = None,
        criteria: Optional[str] = None,
        run_idx: int = 0,
    ) -> Dict[str, Any]:
        """Produce a single mock Decision response."""
        options = options or ["Yes", "No"]
        claim = str(state.get("claim", ""))
        evidence = str(state.get("evidence", ""))

        base_support = self._estimate_semantic_support(claim, evidence)

        # Content hash for deterministic pseudo-random variation
        content_hash = int(hashlib.md5((claim + evidence).encode("utf-8")).hexdigest()[:8], 16)
        run_seed = (content_hash + run_idx * 1013) % (2**31 - 1)
        rng = random.Random(run_seed)

        # Inject minor non-determinism (+/- 0.015)
        jitter = rng.uniform(-0.015, 0.015)
        p_yes = base_support + jitter

        # Round to 0.01 grid as observed in Jev literature
        p_yes = round(min(0.99, max(0.01, p_yes)), 2)
        p_no = round(1.0 - p_yes, 2)

        opt_pos = options[0]
        opt_neg = options[1] if len(options) > 1 else "Other"
        probabilities = {opt_pos: p_yes, opt_neg: p_no}

        chosen = opt_pos if p_yes >= 0.50 else opt_neg

        return {
            "id": f"mock-dec-{int(time.time()*1000)}-{run_idx}",
            "model": "typesafe/jev-1.13",
            "created": int(time.time()),
            "decision_type": "noul",
            "choices": [
                {
                    "index": 0,
                    "decision": chosen,
                    "probabilities": probabilities,
                }
            ],
            "usage": {
                "prompt_tokens": len(claim.split()) + len(evidence.split()) + 35,
                "total_tokens": len(claim.split()) + len(evidence.split()) + 35,
                "cost_usd": 0.00025,
            },
        }
