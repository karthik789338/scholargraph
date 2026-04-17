from pathlib import Path

from src.data.load_scifact import (
    load_raw_scifact,
    normalize_scifact_corpus,
    normalize_scifact_claims,
)
from src.data.normalize import (
    ensure_documents_consistency,
    ensure_queries_consistency,
    filter_empty_documents,
    validate_unique_ids,
)
from src.data.chunk import chunk_documents
from src.graph.schemas import Document, Query, Chunk
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    raw_dir = Path("data/raw/scifact")
    processed_dir = Path("data/processed")

    corpus_path = raw_dir / "corpus.jsonl"
    claims_train_path = raw_dir / "claims_train.jsonl"

    documents_dir = ensure_dir(processed_dir / "documents")
    queries_dir = ensure_dir(processed_dir / "queries")
    chunks_dir = ensure_dir(processed_dir / "chunks")

    logger.info("Loading raw SciFact data...")
    raw_corpus, raw_claims = load_raw_scifact(
        corpus_path=corpus_path,
        claims_path=claims_train_path,
    )

    logger.info("Normalizing documents...")
    docs = normalize_scifact_corpus(raw_corpus)
    docs = ensure_documents_consistency(docs)
    docs = filter_empty_documents(docs)
    validate_unique_ids(docs, "doc_id")

    logger.info(f"Documents after filtering: {len(docs)}")

    logger.info("Normalizing queries...")
    queries = normalize_scifact_claims(raw_claims)
    queries = ensure_queries_consistency(queries)
    validate_unique_ids(queries, "query_id")

    logger.info(f"Queries: {len(queries)}")

    logger.info("Chunking documents...")
    chunks = chunk_documents(
        documents=docs,
        max_tokens=220,
        overlap_tokens=40,
    )
    validate_unique_ids(chunks, "chunk_id")

    logger.info(f"Chunks: {len(chunks)}")

    docs_out = documents_dir / "scifact_documents.jsonl"
    queries_out = queries_dir / "scifact_train.jsonl"
    chunks_out = chunks_dir / "scifact_chunks.jsonl"

    write_jsonl(docs, docs_out)
    write_jsonl(queries, queries_out)
    write_jsonl(chunks, chunks_out)

    logger.info(f"Wrote documents to {docs_out}")
    logger.info(f"Wrote queries to {queries_out}")
    logger.info(f"Wrote chunks to {chunks_out}")


if __name__ == "__main__":
    main()