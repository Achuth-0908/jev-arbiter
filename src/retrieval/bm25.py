"""BM25 evidence indexing and retrieval for atomic claim grounding."""

from __future__ import annotations

import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25SnippetRetriever:
    """Indexes evidence passages and retrieves relevant snippets for claims."""

    def __init__(self, passages: List[str]):
        self.passages = [p.strip() for p in passages if p.strip()]
        if not self.passages:
            self.passages = ["No context available."]
        
        self.corpus_tokens = [simple_tokenize(p) for p in self.passages]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, claim: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """Retrieve top_k passages with BM25 scores for a given claim."""
        query_tokens = simple_tokenize(claim)
        if not query_tokens:
            return [(self.passages[0], 0.0)]

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            # Clamp negative IDF scores to 0.0
            score = max(0.0, float(scores[idx]))
            results.append((self.passages[idx], score))

        return results

    def get_grounding_context(self, claim: str, top_k: int = 2) -> Tuple[str, float]:
        """Return combined top snippet string and highest BM25 score."""
        hits = self.retrieve(claim, top_k=top_k)
        combined_text = "\n".join([snippet for snippet, _ in hits])
        max_score = hits[0][1] if hits else 0.0
        return combined_text, max_score
