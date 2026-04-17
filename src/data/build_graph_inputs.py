from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from src.graph.schemas import GraphInput, Query
from src.retrieval.bm25_index import BM25Index, load_bm25_index, search_bm25
from src.retrieval.dense_index import DenseIndex, batch_search_dense, load_dense_index
from src.retrieval.retrieve import reciprocal_rank_fusion
from src.utils.hashing import make_claim_id
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    records = read_jsonl(path)
    queries = [Query(**record) for record in records]
    logger.info(f"Loaded {len(queries)} queries from {path}")
    return queries


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def get_gold_evidence_chunk_ids(query: Query) -> List[str]:
    return dedupe_preserve_order(
        ev.chunk_id for ev in query.gold_evidence if ev.chunk_id is not None
    )


def get_gold_evidence_doc_ids(query: Query) -> List[str]:
    return dedupe_preserve_order(
        ev.doc_id for ev in query.gold_evidence if ev.doc_id is not None
    )


def build_candidate_claims(query: Query) -> List[Dict[str, Any]]:
    if query.task_type == "claim_verification":
        return [
            {
                "claim_id": make_claim_id(query.query_id, "0"),
                "query_id": query.query_id,
                "doc_id": None,
                "text": query.text,
                "source": "dataset_gold",
                "claim_type": "unknown",
                "confidence": 1.0,
            }
        ]
    return []


def post_filter_results(
    results: Sequence[dict],
    allowed_doc_ids: Optional[Set[str]],
    top_k: int,
) -> List[dict]:
    filtered: List[dict] = []

    for item in results:
        if allowed_doc_ids is not None and item["doc_id"] not in allowed_doc_ids:
            continue
        filtered.append(item)
        if len(filtered) >= top_k:
            break

    for i, item in enumerate(filtered, start=1):
        item["rank"] = i
    return filtered


def run_bm25_search(
    index: BM25Index,
    query_text: str,
    top_k: int,
    allowed_doc_ids: Optional[Set[str]] = None,
    search_multiplier: int = 5,
) -> List[dict]:
    raw = search_bm25(index=index, query=query_text, top_k=max(top_k * search_multiplier, top_k))
    return post_filter_results(raw, allowed_doc_ids, top_k)


def determine_allowed_doc_ids(
    query: Query,
    restrict_to_source_doc: bool = False,
    restrict_to_gold_docs: bool = False,
    restrict_to_candidate_docs: bool = False,
) -> Optional[Set[str]]:
    allowed: Set[str] = set()

    if restrict_to_candidate_docs:
        candidate_doc_ids = getattr(query.metadata, "candidate_doc_ids", [])
        allowed.update(candidate_doc_ids)

    if restrict_to_source_doc and query.source_doc_id:
        allowed.add(query.source_doc_id)

    if restrict_to_gold_docs:
        allowed.update(get_gold_evidence_doc_ids(query))

    return allowed if allowed else None


def build_graph_input_for_query(
    query: Query,
    retrieval_results: Sequence[dict],
) -> GraphInput:
    candidate_chunks = dedupe_preserve_order(item["chunk_id"] for item in retrieval_results)
    candidate_doc_ids = dedupe_preserve_order(item["doc_id"] for item in retrieval_results)
    gold_evidence_chunks = get_gold_evidence_chunk_ids(query)

    retrieval_scores = {
        item["chunk_id"]: {
            "rank": item["rank"],
            "score": float(item.get("score", item.get("rrf_score", 0.0))),
            "doc_id": item["doc_id"],
            "section_title": item.get("section_title"),
        }
        for item in retrieval_results
    }

    graph_input = GraphInput(
        query_id=query.query_id,
        task_type=query.task_type,
        query_text=query.text,
        candidate_chunks=candidate_chunks,
        gold_evidence_chunks=gold_evidence_chunks,
        gold_label=query.gold_label,
        candidate_claims=build_candidate_claims(query),
        metadata={
            "dataset": query.dataset,
            "doc_scope": query.doc_scope,
            "source_doc_id": query.source_doc_id,
            "candidate_doc_ids": candidate_doc_ids,
            "gold_evidence_doc_ids": get_gold_evidence_doc_ids(query),
            "retrieval_scores": retrieval_scores,
            "query_metadata": query.metadata.model_dump(),
        },
    )
    return graph_input


def build_graph_inputs(
    queries: Sequence[Query],
    index_type: str,
    top_k: int,
    bm25_index: Optional[BM25Index] = None,
    dense_index: Optional[DenseIndex] = None,
    restrict_to_source_doc: bool = False,
    restrict_to_gold_docs: bool = False,
    restrict_to_candidate_docs: bool = False,
    model_name_override: Optional[str] = None,
) -> List[GraphInput]:
    graph_inputs: List[GraphInput] = []

    query_texts = [query.text for query in queries]
    allowed_doc_ids_per_query = [
        determine_allowed_doc_ids(
            query=query,
            restrict_to_source_doc=restrict_to_source_doc,
            restrict_to_gold_docs=restrict_to_gold_docs,
            restrict_to_candidate_docs=restrict_to_candidate_docs,
        )
        for query in queries
    ]

    bm25_results_all: Optional[List[List[dict]]] = None
    dense_results_all: Optional[List[List[dict]]] = None

    if index_type in {"bm25", "hybrid"}:
        if bm25_index is None:
            raise ValueError("BM25 index is required for bm25/hybrid mode")

        bm25_results_all = []
        for query_text, allowed_doc_ids in zip(query_texts, allowed_doc_ids_per_query):
            bm25_results_all.append(
                run_bm25_search(
                    index=bm25_index,
                    query_text=query_text,
                    top_k=max(top_k * 3, top_k),
                    allowed_doc_ids=allowed_doc_ids,
                    search_multiplier=1,
                )
            )
        logger.info("Prepared BM25 results for all queries")

    if index_type in {"dense", "hybrid"}:
        if dense_index is None:
            raise ValueError("Dense index is required for dense/hybrid mode")

        dense_results_all = batch_search_dense(
            index=dense_index,
            queries=query_texts,
            top_k=max(top_k * 3, top_k),
            allowed_doc_ids_per_query=allowed_doc_ids_per_query,
            model_name_override=model_name_override,
            batch_size=64,
        )
        logger.info("Prepared dense results for all queries")

    for idx, query in enumerate(queries):
        if index_type == "bm25":
            retrieval_results = post_filter_results(
                bm25_results_all[idx],
                allowed_doc_ids_per_query[idx],
                top_k,
            )

        elif index_type == "dense":
            retrieval_results = post_filter_results(
                dense_results_all[idx],
                allowed_doc_ids_per_query[idx],
                top_k,
            )

        elif index_type == "hybrid":
            retrieval_results = reciprocal_rank_fusion(
                bm25_results_all[idx],
                dense_results_all[idx],
                top_k=top_k,
            )

        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        graph_input = build_graph_input_for_query(
            query=query,
            retrieval_results=retrieval_results,
        )
        graph_inputs.append(graph_input)

        if (idx + 1) % 100 == 0:
            logger.info(f"Built graph inputs for {idx + 1}/{len(queries)} queries")

    logger.info(f"Built {len(graph_inputs)} graph inputs")
    return graph_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph-input bundles from queries and retrieval.")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL")
    parser.add_argument("--output", required=True, help="Path to output graph_inputs JSONL")
    parser.add_argument("--index-type", choices=["bm25", "dense", "hybrid"], required=True)
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--bm25-index-dir", default=None)
    parser.add_argument("--dense-index-dir", default=None)
    parser.add_argument("--model-name-override", default=None)

    parser.add_argument(
        "--restrict-to-source-doc",
        action="store_true",
        help="Filter retrieval to query.source_doc_id when available",
    )
    parser.add_argument(
        "--restrict-to-gold-docs",
        action="store_true",
        help="Filter retrieval to gold evidence docs (useful for debugging only)",
    )
    parser.add_argument(
        "--restrict-to-candidate-docs",
        action="store_true",
        help="Restrict retrieval to SciFact cited candidate docs",
    )

    args = parser.parse_args()

    queries = load_queries(args.queries)

    bm25_index = load_bm25_index(args.bm25_index_dir) if args.bm25_index_dir else None
    dense_index = load_dense_index(args.dense_index_dir) if args.dense_index_dir else None

    graph_inputs = build_graph_inputs(
        queries=queries,
        index_type=args.index_type,
        top_k=args.top_k,
        bm25_index=bm25_index,
        dense_index=dense_index,
        restrict_to_source_doc=args.restrict_to_source_doc,
        restrict_to_gold_docs=args.restrict_to_gold_docs,
        restrict_to_candidate_docs=args.restrict_to_candidate_docs,
        model_name_override=args.model_name_override,
    )

    write_jsonl(graph_inputs, args.output)
    logger.info(f"Wrote graph inputs to {args.output}")


if __name__ == "__main__":
    main()