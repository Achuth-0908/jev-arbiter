"""QA and multi-hop benchmark instance loaders (HotpotQA format)."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class MultiHopQAInstance(BaseModel):
    """Multi-hop QA instance with distractor context passages and agent candidate outputs."""
    id: str
    question: str
    gold_answer: str
    passages: List[str]
    candidate_responses: Dict[str, str]
    candidate_confidences: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


def get_sample_multihop_instances() -> List[MultiHopQAInstance]:
    """Returns representative multi-hop QA instances with agent responses and distractor passages."""
    return [
        MultiHopQAInstance(
            id="hotpot_001",
            question="Which award-winning scientist directed the Manhattan Project laboratory in Los Alamos and what university was he affiliated with?",
            gold_answer="J. Robert Oppenheimer was affiliated with the University of California, Berkeley.",
            passages=[
                "J. Robert Oppenheimer was an American theoretical physicist and director of the Manhattan Project's Los Alamos Laboratory during World War II.",
                "Oppenheimer was a professor of physics at the University of California, Berkeley, where he made significant contributions to theoretical physics.",
                "Ernest Lawrence was an American nuclear scientist and winner of the Nobel Prize in Physics in 1939 for his development of the cyclotron at Berkeley.",
                "The Los Alamos National Laboratory was founded in 1943 as Project Y in New Mexico to coordinate nuclear weapons development.",
            ],
            candidate_responses={
                "agent_claude": "J. Robert Oppenheimer directed the Los Alamos laboratory and was affiliated with the University of California, Berkeley. Confidence: 0.92",
                "agent_llama": "J. Robert Oppenheimer was director of the Los Alamos laboratory and taught physics at UC Berkeley. Confidence: 0.88",
                "agent_gemini": "Ernest Lawrence directed the Los Alamos laboratory during the Manhattan Project and worked at Stanford University. Confidence: 0.80",
                "agent_mistral": "Oppenheimer directed the Los Alamos laboratory during the Manhattan Project, having strong ties to UC Berkeley. Confidence: 0.85",
            },
            candidate_confidences={
                "agent_claude": 0.92,
                "agent_llama": 0.88,
                "agent_gemini": 0.80,
                "agent_mistral": 0.85,
            },
        ),
        MultiHopQAInstance(
            id="hotpot_002",
            question="What programming language created by Bjarne Stroustrup was influenced by Simula, and in what year was it first standardized by ISO?",
            gold_answer="C++ was standardized by ISO in 1998.",
            passages=[
                "C++ is a general-purpose programming language created by Danish computer scientist Bjarne Stroustrup as an extension of the C programming language.",
                "Stroustrup was heavily influenced by Simula 67 while developing C with Classes at Bell Labs.",
                "The first international standard for C++ was ratified by ISO in 1998 as ISO/IEC 14882:1998 (informally known as C++98).",
                "Simula was developed in the 1960s at the Norwegian Computing Center in Oslo by Ole-Johan Dahl and Kristen Nygaard.",
            ],
            candidate_responses={
                "agent_claude": "The language is C++, created by Bjarne Stroustrup with inspiration from Simula, and it was first standardized by ISO in 1998. Confidence: 0.95",
                "agent_llama": "Bjarne Stroustrup designed C++ based on Simula, and the ISO officially standardized it in 1998. Confidence: 0.90",
                "agent_gemini": "C++ was designed by Bjarne Stroustrup and first standardized by ISO in 2011. Confidence: 0.85",
                "agent_mistral": "C++ was developed by Bjarne Stroustrup, influenced by Simula, and achieved ISO standardization in 1998. Confidence: 0.92",
            },
            candidate_confidences={
                "agent_claude": 0.95,
                "agent_llama": 0.90,
                "agent_gemini": 0.85,
                "agent_mistral": 0.92,
            },
        ),
        MultiHopQAInstance(
            id="hotpot_003",
            question="What is the primary function of the ribosome in cellular biology, and does it possess a phospholipid bilayer membrane?",
            gold_answer="Ribosomes synthesize proteins and do not have a phospholipid membrane.",
            passages=[
                "Ribosomes are macromolecular machines found within all living cells that perform biological protein synthesis (mRNA translation).",
                "Unlike eukaryotic organelles such as the nucleus, mitochondria, or chloroplasts, ribosomes are non-membrane-bound complexes composed of RNA and proteins.",
                "The endoplasmic reticulum is a continuous membrane system of folded sacs and tunnels that works closely with ribosomes.",
            ],
            candidate_responses={
                "agent_claude": "Ribosomes perform protein synthesis by translating mRNA. They lack a phospholipid membrane and are non-membrane bound. Confidence: 0.90",
                "agent_llama": "Ribosomes synthesize proteins in cells, and they do not have a surrounding phospholipid bilayer membrane. Confidence: 0.87",
                "agent_gemini": "Ribosomes produce ATP and are enclosed in a double phospholipid membrane. Confidence: 0.78",
                "agent_mistral": "Ribosomes carry out cellular translation of proteins and operate without any phospholipid membrane. Confidence: 0.89",
            },
            candidate_confidences={
                "agent_claude": 0.90,
                "agent_llama": 0.87,
                "agent_gemini": 0.78,
                "agent_mistral": 0.89,
            },
        ),
    ]
