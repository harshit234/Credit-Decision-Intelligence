"""
================================================================================
   HALCYON CREDIT — Policy Retrieval Tool
   Stage 3 | Author: Aditya
   Wraps the existing ChromaDB collection built in Stage 2.
   Queries 'halcyon_policy_v1' with semantic search.
   Falls back to hardcoded conservative rules if ChromaDB unavailable.
================================================================================
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyChunkResult:
    clause_id:    str
    text:         str
    is_hard_stop: bool
    section:      str
    score:        float   # semantic similarity score
    chunk_id:     str


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK — used when ChromaDB is unavailable
# Mirrors the 7 core policies from tools/halcyon_policy_v1.txt
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_POLICIES: list[dict] = [
    {
        "clause_id": "POL-001",
        "text": "Debt-to-income ratio must not exceed 40% for standard loan applicants. "
                "DTI is calculated as (total monthly debt payments × 12) / annual income.",
        "is_hard_stop": True,
        "section": "2.1 Eligibility Criteria",
    },
    {
        "clause_id": "POL-002",
        "text": "Any public record including bankruptcy, tax liens, or court judgements "
                "mandates an automatic DECLINE decision. No exceptions.",
        "is_hard_stop": True,
        "section": "2.2 Hard Stop Conditions",
    },
    {
        "clause_id": "POL-003",
        "text": "Loan-to-income ratio above 3.0 requires senior underwriter review "
                "before approval. Route to REFER.",
        "is_hard_stop": False,
        "section": "2.3 Advisory Flags",
    },
    {
        "clause_id": "POL-004",
        "text": "Debt consolidation purpose combined with DTI above 35% requires "
                "income re-verification before proceeding.",
        "is_hard_stop": False,
        "section": "2.4 Conditional Requirements",
    },
    {
        "clause_id": "POL-005",
        "text": "Thin-file applicants (credit history under 24 months or fewer than "
                "3 open accounts) must be REFERRED for human review. "
                "Automatic decline of thin-file applicants is prohibited.",
        "is_hard_stop": False,
        "section": "2.5 Thin File Policy",
    },
    {
        "clause_id": "POL-006",
        "text": "Two or more delinquencies in the past 24 months triggers mandatory "
                "REFER decision for underwriter review.",
        "is_hard_stop": False,
        "section": "2.6 Delinquency Thresholds",
    },
    {
        "clause_id": "POL-007",
        "text": "Credit score below 580 for non-thin-file applicants mandates automatic "
                "DECLINE. Thin-file applicants with score below 580 must be REFERRED.",
        "is_hard_stop": True,
        "section": "2.2 Hard Stop Conditions",
    },
]


def retrieve_policy(
    query:        str,
    top_k:        int            = 5,
    chroma_path:  Optional[str]  = None,
) -> list[PolicyChunkResult]:
    """
    Retrieve relevant policy clauses for a given underwriting query.

    First tries ChromaDB (Stage 2 vector store).
    Falls back to hardcoded rules if ChromaDB unavailable.

    Args:
        query:       Natural language query describing the applicant situation.
        top_k:       Number of clauses to return.
        chroma_path: Path to ChromaDB persistence dir.

    Returns:
        List of PolicyChunkResult ordered by semantic relevance.
    """
    persist_path = chroma_path or os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=persist_path)
        ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                     model_name="all-MiniLM-L6-v2"
                 )
        collection = client.get_collection(
            name               = "halcyon_policy_v1",
            embedding_function = ef,
        )

        results   = collection.query(
            query_texts = [query],
            n_results   = top_k,
            include     = ["documents", "metadatas", "distances"],
        )

        chunks    = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]
        ids       = results["ids"][0]

        for doc, meta, dist, cid in zip(docs, metas, distances, ids):
            chunks.append(PolicyChunkResult(
                clause_id    = meta.get("clause_id", "UNKNOWN"),
                text         = doc,
                is_hard_stop = bool(meta.get("is_hard_stop", False)),
                section      = meta.get("section", ""),
                score        = round(max(0.0, 1 - dist), 4),
                chunk_id     = cid,
            ))

        return chunks

    except Exception as e:
        print(f"  [PolicyTool] ChromaDB unavailable ({type(e).__name__}: {e}). Using fallback rules.")
        return [
            PolicyChunkResult(
                clause_id    = p["clause_id"],
                text         = p["text"],
                is_hard_stop = p["is_hard_stop"],
                section      = p["section"],
                score        = 1.0,
                chunk_id     = f"fallback-{p['clause_id']}",
            )
            for p in _FALLBACK_POLICIES[:top_k]
        ]
