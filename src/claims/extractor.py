"""Atomic claim extraction and injection sanitization."""

from __future__ import annotations

import re
from typing import List, Optional
from pydantic import BaseModel, Field

from ..clients.llm import UnifiedLLMClient


class AtomicClaim(BaseModel):
    """Represents an atomic, decontextualized factual proposition."""
    claim_text: str
    source_agent_id: str
    self_confidence: float = 0.80
    sanitized: bool = True


class ClaimExtractor:
    """Extracts atomic claims and neutralizes prompt injections."""

    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions?",
        r"(?i)system\s*:\s*you\s+are",
        r"(?i)always\s+(output|answer|choose)\s+(yes|true|option\s+a)",
        r"(?i)override\s+(prompt|system|evaluator)",
        r"(?i)do\s+not\s+verify",
    ]

    def __init__(self, llm_client: Optional[UnifiedLLMClient] = None):
        self.llm = llm_client or UnifiedLLMClient()

    def sanitize_text(self, text: str) -> str:
        """Strip suspected adversarial instruction injections from text."""
        cleaned = text
        for pat in self.INJECTION_PATTERNS:
            cleaned = re.sub(pat, "[FILTERED_INJECTION]", cleaned)
        return cleaned

    def extract_heuristic(self, text: str, agent_id: str, self_confidence: float = 0.80) -> List[AtomicClaim]:
        """Fast offline heuristic sentence-level claim segmenter."""
        sanitized = self.sanitize_text(text)
        # Remove verbalized confidence expressions
        sanitized = re.sub(r"(?i)confidence\s*[:=-]\s*([0-1]?\.\d+|\d+%)", "", sanitized)
        # Split by periods, semicolons, or bullet points
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+|- |\* ", sanitized)
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Clean leading numbering or bullets
            s_clean = re.sub(r"^\d+[\.\)]\s*", "", s_clean).strip()
            if len(s_clean.split()) >= 4 and not s_clean.lower().startswith("confidence:"):
                claims.append(
                    AtomicClaim(
                        claim_text=s_clean,
                        source_agent_id=agent_id,
                        self_confidence=self_confidence,
                        sanitized=True,
                    )
                )
        return claims

    async def extract(
        self,
        text: str,
        agent_id: str,
        self_confidence: float = 0.80,
        model: str = "meta-llama/llama-3.1-8b-instruct",
    ) -> List[AtomicClaim]:
        """Extract atomic claims using LLM or offline fallback."""
        sanitized = self.sanitize_text(text)

        # When running in mock mode, use deterministic heuristic extraction
        if self.llm.mock_mode:
            return self.extract_heuristic(sanitized, agent_id, self_confidence)

        prompt = (
            "Extract all atomic, verifiable factual claims from the response below. "
            "Output each claim on a new line prefixed with '- '.\n\n"
            f"Response:\n{sanitized}\n\nClaims:"
        )

        try:
            resp = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
            )
            extracted_lines = [
                line.strip().lstrip("-* ").strip()
                for line in resp.content.split("\n")
                if len(line.strip().lstrip("-* ").strip().split()) >= 3
            ]
            if extracted_lines:
                return [
                    AtomicClaim(
                        claim_text=line,
                        source_agent_id=agent_id,
                        self_confidence=self_confidence,
                    )
                    for line in extracted_lines
                ]
        except Exception:
            pass

        return self.extract_heuristic(sanitized, agent_id, self_confidence)
