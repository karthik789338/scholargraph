from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.eval.qasper_metrics import compute_evidence_metrics
from src.graph.schemas import GraphInput, Query
from src.retrieval.dense_index import encode_queries, load_dense_index
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_graph_inputs(path: str | Path) -> List[GraphInput]:
    return [GraphInput(**x) for x in read_jsonl(path)]


def load_local_graphs(path: str | Path) -> List[dict]:
    return read_jsonl(path)


def map_local_graphs_by_query_id(local_graphs: Sequence[dict]) -> Dict[str, dict]:
    return {graph["query_id"]: graph for graph in local_graphs}


def evidence_id_to_chunk_id(evidence_id: str) -> str:
    prefix = "evidence::"
    if evidence_id.startswith(prefix):
        return evidence_id[len(prefix):]
    return evidence_id


def extract_evidence_degree_weights(local_graph: dict) -> Dict[str, float]:
    """
    Graph-only evidence weights from evidence<->evidence edges.
    Avoids claim->evidence edges to reduce risk of supervision leakage.
    """
    evidence_nodes = local_graph.get("evidence", [])
    edges = local_graph.get("edges", [])

    chunk_ids = []
    for node in evidence_nodes:
        chunk_id = node.get("chunk_id")
        if chunk_id:
            chunk_ids.append(chunk_id)

    if not chunk_ids:
        return {}

    degree_scores = {chunk_id: 0.0 for chunk_id in chunk_ids}

    for edge in edges:
        if edge.get("src_type") == "evidence" and edge.get("dst_type") == "evidence":
            src_chunk = evidence_id_to_chunk_id(edge["src_id"])
            dst_chunk = evidence_id_to_chunk_id(edge["dst_id"])
            score = float(edge.get("score", 1.0))

            if src_chunk in degree_scores:
                degree_scores[src_chunk] += score
            if dst_chunk in degree_scores:
                degree_scores[dst_chunk] += score

    max_degree = max(degree_scores.values()) if degree_scores else 0.0
    if max_degree <= 0:
        return {chunk_id: 0.0 for chunk_id in degree_scores}

    return {
        chunk_id: degree / max_degree
        for chunk_id, degree in degree_scores.items()
    }


def build_dense_chunk_lookup(dense_index) -> Dict[str, int]:
    return {chunk.chunk_id: idx for idx, chunk in enumerate(dense_index.chunks)}


def normalize_retrieval_priors(graph_input: GraphInput) -> Dict[str, float]:
    """
    Retrieval prior based on rank and score, normalized to [0, 1].
    """
    retrieval_scores = graph_input.metadata.get("retrieval_scores", {})
    priors = {}

    for chunk_id in graph_input.candidate_chunks:
        info = retrieval_scores.get(chunk_id, {})
        rank = int(info.get("rank", 10**9))
        score = float(info.get("score", 0.0))

        # rank prior + small score bonus
        prior = (1.0 / max(rank, 1)) + 0.05 * score
        priors[chunk_id] = prior

    max_prior = max(priors.values()) if priors else 0.0
    if max_prior <= 0:
        return {cid: 0.0 for cid in priors}

    return {cid: p / max_prior for cid, p in priors.items()}


def cosine_to_unit_interval(x: float) -> float:
    """
    Convert cosine similarity [-1, 1] to [0, 1].
    """
    return (x + 1.0) / 2.0


def compute_question_aware_scores(
    graph_input: GraphInput,
    local_graph: dict,
    query_embedding: np.ndarray,
    dense_index,
    chunk_id_to_dense_idx: Dict[str, int],
    semantic_weight: float = 0.55,
    retrieval_weight: float = 0.25,
    graph_weight: float = 0.20,
) -> Dict[str, Dict[str, float]]:
    """
    Per chunk:
    final_score = w_sem * semantic_relevance
                + w_ret * retrieval_prior
                + w_graph * graph_degree_weight
    """
    retrieval_priors = normalize_retrieval_priors(graph_input)
    graph_weights = extract_evidence_degree_weights(local_graph)

    chunk_scores: Dict[str, Dict[str, float]] = {}

    for chunk_id in graph_input.candidate_chunks:
        dense_idx = chunk_id_to_dense_idx.get(chunk_id)
        if dense_idx is None:
            continue

        chunk_embedding = dense_index.embeddings[dense_idx]
        cosine_sim = float(np.dot(query_embedding, chunk_embedding))
        semantic_relevance = cosine_to_unit_interval(cosine_sim)

        retrieval_prior = retrieval_priors.get(chunk_id, 0.0)
        graph_degree = graph_weights.get(chunk_id, 0.0)

        final_score = (
            semantic_weight * semantic_relevance
            + retrieval_weight * retrieval_prior
            + graph_weight * graph_degree
        )

        chunk_scores[chunk_id] = {
            "semantic_relevance": semantic_relevance,
            "retrieval_prior": retrieval_prior,
            "graph_degree": graph_degree,
            "final_score": final_score,
        }

    return chunk_scores


def mmr_select_chunks(
    candidate_chunk_ids: Sequence[str],
    chunk_scores: Dict[str, Dict[str, float]],
    dense_index,
    chunk_id_to_dense_idx: Dict[str, int],
    top_k: int = 3,
    mmr_lambda: float = 0.75,
) -> List[str]:
    """
    MMR selection to avoid redundant top-3 evidence chunks.
    """
    selected: List[str] = []
    remaining = [cid for cid in candidate_chunk_ids if cid in chunk_scores]

    if not remaining:
        return selected

    while remaining and len(selected) < top_k:
        best_chunk = None
        best_score = float("-inf")

        for cid in remaining:
            relevance = chunk_scores[cid]["final_score"]

            if not selected:
                mmr_score = relevance
            else:
                cid_idx = chunk_id_to_dense_idx[cid]
                cid_emb = dense_index.embeddings[cid_idx]

                redundancy = max(
                    cosine_to_unit_interval(float(np.dot(cid_emb, dense_index.embeddings[chunk_id_to_dense_idx[sid]])))
                    for sid in selected
                )
                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_chunk = cid

        if best_chunk is None:
            break

        selected.append(best_chunk)
        remaining.remove(best_chunk)

    return selected


def build_predictions(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    local_graphs_by_query_id: Dict[str, dict],
    dense_index,
    top_k: int = 3,
    graph_beta: float = 0.50,  # kept for backward compatibility in CLI naming
    semantic_weight: float = 0.55,
    retrieval_weight: float = 0.25,
    graph_weight: float = 0.20,
    mmr_lambda: float = 0.75,
) -> List[Dict]:
    """
    Question-aware graph reranker.
    """
    del graph_beta  # replaced conceptually by graph_weight

    query_texts = [q.text for q in queries]
    query_id_to_embedding = {}

    query_embeddings = encode_queries(
        queries=query_texts,
        model_name=dense_index.model_name,
        batch_size=64,
    )

    for query, emb in zip(queries, query_embeddings):
        query_id_to_embedding[query.query_id] = emb

    chunk_id_to_dense_idx = build_dense_chunk_lookup(dense_index)

    predictions = []

    for idx, graph_input in enumerate(graph_inputs, start=1):
        local_graph = local_graphs_by_query_id.get(graph_input.query_id, {})
        query_embedding = query_id_to_embedding[graph_input.query_id]

        chunk_scores = compute_question_aware_scores(
            graph_input=graph_input,
            local_graph=local_graph,
            query_embedding=query_embedding,
            dense_index=dense_index,
            chunk_id_to_dense_idx=chunk_id_to_dense_idx,
            semantic_weight=semantic_weight,
            retrieval_weight=retrieval_weight,
            graph_weight=graph_weight,
        )

        ranked_chunk_ids = sorted(
            graph_input.candidate_chunks,
            key=lambda cid: chunk_scores.get(cid, {}).get("final_score", float("-inf")),
            reverse=True,
        )

        pred_chunks = mmr_select_chunks(
            candidate_chunk_ids=ranked_chunk_ids,
            chunk_scores=chunk_scores,
            dense_index=dense_index,
            chunk_id_to_dense_idx=chunk_id_to_dense_idx,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
        )

        predictions.append(
            {
                "query_id": graph_input.query_id,
                "predicted_evidence_chunks": pred_chunks,
                "candidate_chunk_count": len(graph_input.candidate_chunks),
                "scored_chunk_count": min(top_k, len(graph_input.candidate_chunks)),
                "semantic_weight": semantic_weight,
                "retrieval_weight": retrieval_weight,
                "graph_weight": graph_weight,
                "mmr_lambda": mmr_lambda,
                "top_chunk_scores": [
                    {
                        "chunk_id": cid,
                        **chunk_scores[cid],
                    }
                    for cid in ranked_chunk_ids[:5]
                    if cid in chunk_scores
                ],
            }
        )

        if idx % 500 == 0 or idx == len(graph_inputs):
            logger.info(f"Built question-aware QASPER predictions for {idx}/{len(graph_inputs)} queries")

    logger.info(f"Built {len(predictions)} QASPER question-aware graph reranking predictions")
    return predictions


def evaluate_predictions(
    queries: Sequence[Query],
    predictions: Sequence[Dict],
) -> Dict:
    pred_map = {p["query_id"]: p for p in predictions}

    gold_sets = []
    pred_sets = []

    for query in queries:
        gold = {
            ev.chunk_id
            for ev in query.gold_evidence
            if ev.chunk_id is not None
        }
        pred = set(pred_map.get(query.query_id, {}).get("predicted_evidence_chunks", []))

        gold_sets.append(gold)
        pred_sets.append(pred)

    return compute_evidence_metrics(gold_sets, pred_sets)


def main():
    parser = argparse.ArgumentParser(description="Run question-aware QASPER graph reranker")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--graph-inputs", required=True)
    parser.add_argument("--local-graphs", required=True)
    parser.add_argument("--dense-index-dir", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--semantic-weight", type=float, default=0.55)
    parser.add_argument("--retrieval-weight", type=float, default=0.25)
    parser.add_argument("--graph-weight", type=float, default=0.20)
    parser.add_argument("--mmr-lambda", type=float, default=0.75)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    graph_inputs = load_graph_inputs(args.graph_inputs)
    local_graphs = load_local_graphs(args.local_graphs)
    local_graphs_by_query_id = map_local_graphs_by_query_id(local_graphs)
    dense_index = load_dense_index(args.dense_index_dir)

    predictions = build_predictions(
        queries=queries,
        graph_inputs=graph_inputs,
        local_graphs_by_query_id=local_graphs_by_query_id,
        dense_index=dense_index,
        top_k=args.top_k,
        semantic_weight=args.semantic_weight,
        retrieval_weight=args.retrieval_weight,
        graph_weight=args.graph_weight,
        mmr_lambda=args.mmr_lambda,
    )
    write_jsonl(predictions, args.output_predictions)

    metrics = evaluate_predictions(
        queries=queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote predictions to {args.output_predictions}")
    logger.info(f"Wrote metrics to {args.output_metrics}")
    logger.info(f"Evidence micro F1: {metrics['micro_f1']:.4f}")


if __name__ == "__main__":
    main()