from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.retrieval.bm25_index import load_bm25_index, search_bm25
from src.retrieval.dense_index import load_dense_index, search_dense
from src.utils.logging import get_logger

logger = get_logger(__name__)


def retrieve_bm25(
    index_dir: str | Path,
    query: str,
    top_k: int = 10,
    doc_id: Optional[str] = None,
) -> List[dict]:
    index = load_bm25_index(index_dir)
    return search_bm25(index=index, query=query, top_k=top_k, doc_id=doc_id)


def retrieve_dense(
    index_dir: str | Path,
    query: str,
    top_k: int = 10,
    doc_id: Optional[str] = None,
    model_name_override: Optional[str] = None,
) -> List[dict]:
    index = load_dense_index(index_dir)
    return search_dense(
        index=index,
        query=query,
        top_k=top_k,
        doc_id=doc_id,
        model_name_override=model_name_override,
    )


def reciprocal_rank_fusion(
    results_a: List[dict],
    results_b: List[dict],
    k: int = 60,
    top_k: int = 10,
) -> List[dict]:
    """
    Simple hybrid retriever using Reciprocal Rank Fusion.
    """
    fused = {}

    for results in [results_a, results_b]:
        for item in results:
            chunk_id = item["chunk_id"]
            rank = item["rank"]
            fused.setdefault(
                chunk_id,
                {
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "section_id": item["section_id"],
                    "section_title": item["section_title"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                },
            )
            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx

    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve chunks using BM25 or dense index.")
    parser.add_argument(
        "--index-type",
        choices=["bm25", "dense", "hybrid"],
        required=True,
        help="Retriever type",
    )
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--doc-id", default=None, help="Optional document filter")

    parser.add_argument("--bm25-index-dir", default=None, help="BM25 index directory")
    parser.add_argument("--dense-index-dir", default=None, help="Dense index directory")
    parser.add_argument(
        "--model-name-override",
        default=None,
        help="Optional dense model override at query time",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON",
    )

    args = parser.parse_args()

    if args.index_type == "bm25":
        if not args.bm25_index_dir:
            raise ValueError("--bm25-index-dir is required for BM25 retrieval")
        results = retrieve_bm25(
            index_dir=args.bm25_index_dir,
            query=args.query,
            top_k=args.top_k,
            doc_id=args.doc_id,
        )

    elif args.index_type == "dense":
        if not args.dense_index_dir:
            raise ValueError("--dense-index-dir is required for dense retrieval")
        results = retrieve_dense(
            index_dir=args.dense_index_dir,
            query=args.query,
            top_k=args.top_k,
            doc_id=args.doc_id,
            model_name_override=args.model_name_override,
        )

    else:
        if not args.bm25_index_dir or not args.dense_index_dir:
            raise ValueError("--bm25-index-dir and --dense-index-dir are required for hybrid retrieval")

        bm25_results = retrieve_bm25(
            index_dir=args.bm25_index_dir,
            query=args.query,
            top_k=max(args.top_k * 3, 20),
            doc_id=args.doc_id,
        )
        dense_results = retrieve_dense(
            index_dir=args.dense_index_dir,
            query=args.query,
            top_k=max(args.top_k * 3, 20),
            doc_id=args.doc_id,
            model_name_override=args.model_name_override,
        )
        results = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=args.top_k,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved retrieval results to {output_path}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()