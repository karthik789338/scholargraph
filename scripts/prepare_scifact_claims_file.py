from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_scifact import (
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
from src.utils.io import ensure_dir, write_jsonl, read_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SciFact queries from an arbitrary claims JSONL file.")
    parser.add_argument("--claims-path", required=True, help="Path to raw SciFact claims JSONL")
    parser.add_argument("--output-name", required=True, help="Output query stem, e.g. scifact_fold0_train")
    parser.add_argument("--corpus-path", default="data/raw/scifact/corpus.jsonl")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument(
        "--force-rebuild-corpus",
        action="store_true",
        help="Rebuild normalized docs/chunks even if they already exist",
    )
    args = parser.parse_args()

    claims_path = Path(args.claims_path)
    corpus_path = Path(args.corpus_path)

    processed_dir = Path("data/processed")
    documents_dir = ensure_dir(processed_dir / "documents")
    queries_dir = ensure_dir(processed_dir / "queries")
    chunks_dir = ensure_dir(processed_dir / "chunks")

    docs_out = documents_dir / "scifact_documents.jsonl"
    chunks_out = chunks_dir / "scifact_chunks.jsonl"
    queries_out = queries_dir / f"{args.output_name}.jsonl"

    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    raw_claims = read_jsonl(claims_path)
    if not raw_claims:
        raise ValueError(f"No claims found in {claims_path}")

    logger.info(f"Normalizing queries from {claims_path} ...")
    queries = normalize_scifact_claims(raw_claims)
    queries = ensure_queries_consistency(queries)
    validate_unique_ids(queries, "query_id")
    write_jsonl(queries, queries_out)
    logger.info(f"Wrote {len(queries)} queries to {queries_out}")

    if chunks_out.exists() and docs_out.exists() and not args.force_rebuild_corpus:
        logger.info("Chunks/documents already exist; skipping corpus rebuild.")
        return

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    raw_corpus = read_jsonl(corpus_path)
    if not raw_corpus:
        raise ValueError(f"No corpus documents found in {corpus_path}")

    logger.info("Normalizing documents...")
    docs = normalize_scifact_corpus(raw_corpus)
    docs = ensure_documents_consistency(docs)
    docs = filter_empty_documents(docs)
    validate_unique_ids(docs, "doc_id")
    logger.info(f"Documents after filtering: {len(docs)}")

    logger.info("Chunking documents...")
    chunks = chunk_documents(
        documents=docs,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    validate_unique_ids(chunks, "chunk_id")
    logger.info(f"Chunks: {len(chunks)}")

    write_jsonl(docs, docs_out)
    write_jsonl(chunks, chunks_out)
    logger.info(f"Wrote documents to {docs_out}")
    logger.info(f"Wrote chunks to {chunks_out}")


if __name__ == "__main__":
    main()
