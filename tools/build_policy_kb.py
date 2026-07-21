"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HALCYON CREDIT — Policy Knowledge Base Builder                           ║
║   Stage 2 | Author: Himkar                                                 ║
║   Ingests Halcyon lending policy into ChromaDB for RAG retrieval           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Outputs:
  chroma_db/halcyon_policy_v1/   → persistent ChromaDB collection on disk

Usage:
  python build_policy_kb.py              # build and smoke-test
  python build_policy_kb.py --rebuild    # wipe and rebuild from scratch
"""

import os
import sys
import re
import argparse
import shutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
POLICY_DOC_PATH  = "halcyon_policy_v1.txt"
CHROMA_DIR       = "../chroma_db"
COLLECTION_NAME  = "halcyon_policy_v1"
EMBED_MODEL      = "all-MiniLM-L6-v2"   # 384-dim, fast, runs offline

# Smoke test queries → expected clause retrievals
SMOKE_TESTS = [
    {
        "query"    : "applicant has high debt to income ratio above threshold",
        "expected" : "POL-001",
    },
    {
        "query"    : "applicant has bankruptcy or public record on file",
        "expected" : "POL-002",
    },
    {
        "query"    : "thin file applicant with no credit history very short",
        "expected" : "POL-005",
    },
    {
        "query"    : "applicant credit score is below 580 minimum requirement",
        "expected" : "POL-007",
    },
    {
        "query"    : "debt consolidation loan with elevated dti requires re-verification",
        "expected" : "POL-004",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PARSE POLICY DOCUMENT INTO CLAUSE CHUNKS
# ─────────────────────────────────────────────────────────────────────────────
def parse_policy_document(path: str) -> list[dict]:
    """
    Parses the flat policy text file into structured clause chunks.
    Each chunk maps to one policy clause with rich metadata.
    """
    console.rule("[bold cyan]Section 1 — Parsing Policy Document[/]")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on clause boundaries (===... CLAUSE POL-XXX ...)
    clause_pattern = re.compile(
        r"={10,}\nCLAUSE (POL-\d+) — (.+?) \((.+?)\)\n={10,}(.*?)(?=\n={10,}|\nEND OF)",
        re.DOTALL
    )
    matches = clause_pattern.findall(raw)

    clauses = []
    for clause_id, title, stop_label, body in matches:
        # Extract metadata from body
        is_hard_stop  = "YES" in stop_label or "HARD STOP" in stop_label
        threshold_match = re.search(r"Threshold\s+:\s+(.+)", body)
        threshold_val   = threshold_match.group(1).strip() if threshold_match else ""
        section_match   = re.search(r"Category\s+:\s+(.+)", body)
        section_val     = section_match.group(1).strip() if section_match else ""

        # Build the document text to embed — full body gives best retrieval
        doc_text = f"Clause {clause_id} — {title}\n{body.strip()}"

        clauses.append({
            "clause_id"   : clause_id.strip(),
            "title"       : title.strip(),
            "is_hard_stop": is_hard_stop,
            "threshold"   : threshold_val,
            "section"     : section_val,
            "text"        : doc_text,
        })

        status = "[red]HARD STOP[/]" if is_hard_stop else "[green]Advisory[/]"
        console.print(f"  Parsed {clause_id.strip():8s} | {status:20s} | {title.strip()[:45]}")

    console.print(f"\n  Total clauses parsed: [bold]{len(clauses)}[/]")
    return clauses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BUILD CHROMADB COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def build_collection(clauses: list[dict], rebuild: bool = False):
    console.rule("[bold cyan]Section 2 — Building ChromaDB Collection[/]")

    import chromadb
    from chromadb.utils import embedding_functions

    # Wipe existing if rebuild requested
    collection_path = os.path.join(CHROMA_DIR, COLLECTION_NAME)
    if rebuild and os.path.exists(collection_path):
        shutil.rmtree(collection_path)
        console.print("  [yellow]Existing collection wiped for rebuild.[/]")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Persistent client — stores to disk
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Sentence-transformer embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    console.print(f"  Embedding model  : {EMBED_MODEL}")

    # Get or create collection
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef
        )
        console.print(f"  Collection exists with {collection.count()} docs — skipping ingest.")
        console.print("  Run with --rebuild to force re-ingest.")
        return client, collection
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

    # Ingest all clauses
    console.print(f"  Ingesting {len(clauses)} clauses into ChromaDB...")
    collection.add(
        ids        = [c["clause_id"] for c in clauses],
        documents  = [c["text"] for c in clauses],
        metadatas  = [
            {
                "clause_id"   : c["clause_id"],
                "title"       : c["title"],
                "is_hard_stop": str(c["is_hard_stop"]),
                "threshold"   : c["threshold"],
                "section"     : c["section"],
            }
            for c in clauses
        ],
    )
    console.print(f"  [bold green]OK: {collection.count()} clauses stored in ChromaDB[/]")
    console.print(f"  Persisted at: {CHROMA_DIR}/")
    return client, collection


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SMOKE TEST (RAG RETRIEVAL VALIDATION)
# ─────────────────────────────────────────────────────────────────────────────
def run_smoke_tests(collection):
    console.rule("[bold cyan]Section 3 — Smoke Test: RAG Retrieval[/]")
    console.print("  Running 5 test queries to validate retrieval quality...\n")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Query", style="white", max_width=40)
    table.add_column("Expected", style="cyan", justify="center")
    table.add_column("Top Hit", style="yellow", justify="center")
    table.add_column("Result", justify="center")

    passed = 0
    for test in SMOKE_TESTS:
        results = collection.query(
            query_texts=[test["query"]],
            n_results=2,
            include=["metadatas", "distances"]
        )
        top_hit  = results["metadatas"][0][0]["clause_id"]
        distance = results["distances"][0][0]
        correct  = top_hit == test["expected"]
        if correct:
            passed += 1
            status = "[bold green]PASS[/]"
        else:
            status = f"[bold red]FAIL (got {top_hit})[/]"

        table.add_row(
            test["query"][:40],
            test["expected"],
            f"{top_hit} (dist={distance:.3f})",
            status,
        )

    console.print(table)
    console.print(f"\n  Score: [bold]{passed}/{len(SMOKE_TESTS)}[/] smoke tests passed")

    if passed == len(SMOKE_TESTS):
        console.print(Panel(
            "[bold green]PASS: Policy Knowledge Base is READY[/]\n\n"
            "All retrieval tests passed. The PolicyCompliantAgent can now\n"
            "query ChromaDB to fetch relevant clauses for any loan application.",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]WARN: {len(SMOKE_TESTS) - passed} test(s) failed.\n"
            "Check embedding model or policy document parsing.",
            border_style="yellow"
        ))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe and rebuild the ChromaDB collection from scratch")
    args = parser.parse_args()

    clauses = parse_policy_document(POLICY_DOC_PATH)
    client, collection = build_collection(clauses, rebuild=args.rebuild)
    run_smoke_tests(collection)
