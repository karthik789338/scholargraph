from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SciFact data for a specific split.")
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        required=True,
        help="Which SciFact claim split to prepare",
    )
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    args = parser.parse_args()

    raw_dir = Path("data/raw/scifact")
    processed_dir = Path("data/processed")

    corpus_path = raw_dir / "corpus.jsonl"
    claims_path = raw_dir / f"claims_{args.split}.jsonl"

    documents_dir = ensure_dir(processed_dir / "documents")
    queries_dir = ensure_dir(processed_dir / "queries")
    chunks_dir = ensure_dir(processed_dir / "chunks")

    logger.info(f"Loading raw SciFact data for split={args.split}...")
    raw_corpus, raw_claims = load_raw_scifact(
        corpus_path=corpus_path,
        claims_path=claims_path,
    )

    if not raw_corpus:
        raise ValueError(
            f"No SciFact corpus documents were loaded from {corpus_path}. "
            f"Check that the file exists, is non-empty, and contains JSONL records."
        )

    if not raw_claims:
        raise ValueError(
            f"No SciFact claims were loaded from {claims_path}. "
            f"Check that the file exists, is non-empty, and contains JSONL records."
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

    # Chunks are corpus-level, so this will reproduce the same chunk file each time.
    logger.info("Chunking documents...")
    chunks = chunk_documents(
        documents=docs,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    validate_unique_ids(chunks, "chunk_id")
    logger.info(f"Chunks: {len(chunks)}")

    docs_out = documents_dir / "scifact_documents.jsonl"
    queries_out = queries_dir / f"scifact_{args.split}.jsonl"
    chunks_out = chunks_dir / "scifact_chunks.jsonl"

    write_jsonl(docs, docs_out)
    write_jsonl(queries, queries_out)
    write_jsonl(chunks, chunks_out)

    logger.info(f"Wrote documents to {docs_out}")
    logger.info(f"Wrote queries to {queries_out}")
    logger.info(f"Wrote chunks to {chunks_out}")


if __name__ == "__main__":
    main()
