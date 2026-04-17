from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunk import chunk_documents
from src.data.load_scifact_open import (
    load_scifact_open_claims,
    load_scifact_open_documents,
    normalize_scifact_open_documents,
    normalize_scifact_open_queries,
)
from src.data.normalize import (
    ensure_documents_consistency,
    ensure_queries_consistency,
    filter_empty_documents,
    validate_unique_ids,
)
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    claims_path = Path("data/raw/scifact_open/claims.jsonl")
    docs_path = Path("data/raw/scifact_open/documents.jsonl")

    processed_dir = Path("data/processed")
    docs_out_dir = ensure_dir(processed_dir / "documents")
    queries_out_dir = ensure_dir(processed_dir / "queries")
    chunks_out_dir = ensure_dir(processed_dir / "chunks")

    logger.info("Loading SciFact-Open raw data...")
    raw_claims = load_scifact_open_claims(claims_path)
    raw_docs = load_scifact_open_documents(docs_path)

    logger.info("Normalizing documents...")
    docs = normalize_scifact_open_documents(raw_docs)
    docs = ensure_documents_consistency(docs)
    docs = filter_empty_documents(docs)
    validate_unique_ids(docs, "doc_id")
    logger.info(f"Documents after filtering: {len(docs)}")

    logger.info("Normalizing queries...")
    queries = normalize_scifact_open_queries(raw_claims)
    queries = ensure_queries_consistency(queries)
    validate_unique_ids(queries, "query_id")
    logger.info(f"Queries: {len(queries)}")

    logger.info("Chunking documents...")
    chunks = chunk_documents(docs, max_tokens=220, overlap_tokens=40)
    validate_unique_ids(chunks, "chunk_id")
    logger.info(f"Chunks: {len(chunks)}")

    docs_out = docs_out_dir / "scifact_open_documents.jsonl"
    queries_out = queries_out_dir / "scifact_open.jsonl"
    chunks_out = chunks_out_dir / "scifact_open_chunks.jsonl"

    write_jsonl(docs, docs_out)
    write_jsonl(queries, queries_out)
    write_jsonl(chunks, chunks_out)

    logger.info(f"Wrote documents to {docs_out}")
    logger.info(f"Wrote queries to {queries_out}")
    logger.info(f"Wrote chunks to {chunks_out}")


if __name__ == "__main__":
    main()