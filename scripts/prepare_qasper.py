from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_qasper import load_raw_qasper, normalize_qasper_papers, normalize_qasper_queries
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


def main():
    parser = argparse.ArgumentParser(description="Prepare QASPER split")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    args = parser.parse_args()

    raw_path = Path(f"data/raw/qasper/{args.split}.json")
    processed_dir = Path("data/processed")

    documents_dir = ensure_dir(processed_dir / "documents")
    queries_dir = ensure_dir(processed_dir / "queries")
    chunks_dir = ensure_dir(processed_dir / "chunks")

    logger.info(f"Loading raw QASPER split: {args.split}")
    raw = load_raw_qasper(raw_path)

    logger.info("Normalizing documents...")
    docs = normalize_qasper_papers(raw)
    docs = ensure_documents_consistency(docs)
    docs = filter_empty_documents(docs)
    validate_unique_ids(docs, "doc_id")
    logger.info(f"Documents after filtering: {len(docs)}")

    logger.info("Normalizing queries...")
    queries = normalize_qasper_queries(raw)
    queries = ensure_queries_consistency(queries)
    validate_unique_ids(queries, "query_id")
    logger.info(f"Queries: {len(queries)}")

    logger.info("Chunking documents...")
    chunks = chunk_documents(
        documents=docs,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    validate_unique_ids(chunks, "chunk_id")
    logger.info(f"Chunks: {len(chunks)}")

    docs_out = documents_dir / f"qasper_{args.split}_documents.jsonl"
    queries_out = queries_dir / f"qasper_{args.split}.jsonl"
    chunks_out = chunks_dir / f"qasper_{args.split}_chunks.jsonl"

    write_jsonl(docs, docs_out)
    write_jsonl(queries, queries_out)
    write_jsonl(chunks, chunks_out)

    logger.info(f"Wrote documents to {docs_out}")
    logger.info(f"Wrote queries to {queries_out}")
    logger.info(f"Wrote chunks to {chunks_out}")


if __name__ == "__main__":
    main()