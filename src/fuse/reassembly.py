"""Reassembly of verified atomic claims into a coherent answer."""

from __future__ import annotations

from typing import List


def reassemble_answer_from_claims(
    accepted_claims: List[str],
    query: str = "",
    synthesis_style: str = "concise",
) -> str:
    """Compose the final response strictly from verified, accepted claims.
    
    This enforces the invariant that ungrounded / refuted hallucinated claims
    are entirely excluded from the synthesized answer.
    """
    if not accepted_claims:
        return "The available evidence does not sufficiently support any affirmative answer to this query (Abstention)."

    if synthesis_style == "bullet":
        lines = [f"- {claim}" for claim in accepted_claims]
        return "\n".join(lines)

    # Coherent paragraph compilation
    cleaned_claims = []
    for c in accepted_claims:
        c_clean = c.strip().rstrip(".")
        if c_clean:
            cleaned_claims.append(c_clean)

    return ". ".join(cleaned_claims) + "."
