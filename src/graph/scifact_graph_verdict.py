from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.baselines.scifact_baseline import (
    score_claim_evidence_pairs,
    select_top_chunks_for_nli,
)
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.schemas import Chunk, GraphInput, Query
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_graph_inputs(path: str | Path) -> List[GraphInput]:
    return [GraphInput(**x) for x in read_jsonl(path)]


def load_chunks(path: str | Path) -> List[Chunk]:
    return [Chunk(**x) for x in read_jsonl(path)]


def load_local_graphs(path: str | Path) -> List[dict]:
    return read_jsonl(path)


def map_chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def map_local_graphs_by_query_id(local_graphs: Sequence[dict]) -> Dict[str, dict]:
    return {graph["query_id"]: graph for graph in local_graphs}


def evidence_id_to_chunk_id(evidence_id: str) -> str:
    """
    evidence::chunk_id -> chunk_id
    """
    prefix = "evidence::"
    if evidence_id.startswith(prefix):
        return evidence_id[len(prefix):]
    return evidence_id


def extract_evidence_degree_weights(local_graph: dict) -> Dict[str, float]:
    """
    Build graph-aware weights from evidence<->evidence edges only.
    This avoids using potentially gold-dependent claim->evidence relations.
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


def build_graph_node_weights(
    graph_input: GraphInput,
    local_graph: dict,
    selected_chunk_ids: Sequence[str],
) -> Dict[str, float]:
    """
    Weight each chunk by:
    - retrieval rank
    - graph connectivity (degree over evidence-evidence edges)
    """
    retrieval_scores = graph_input.metadata.get("retrieval_scores", {})
    degree_weights = extract_evidence_degree_weights(local_graph)

    raw_weights: Dict[str, float] = {}

    for chunk_id in selected_chunk_ids:
        info = retrieval_scores.get(chunk_id, {})
        rank = int(info.get("rank", 999999))

        rank_weight = 1.0 / max(rank, 1)
        graph_weight = 1.0 + 0.5 * degree_weights.get(chunk_id, 0.0)

        raw_weights[chunk_id] = rank_weight * graph_weight

    total = sum(raw_weights.values())
    if total <= 0:
        return {chunk_id: 1.0 / len(selected_chunk_ids) for chunk_id in selected_chunk_ids} if selected_chunk_ids else {}

    return {
        chunk_id: weight / total
        for chunk_id, weight in raw_weights.items()
    }


def aggregate_graph_scores(
    scored_chunks: Sequence[Dict],
    node_weights: Dict[str, float],
) -> Dict[str, float]:
    support = 0.0
    refute = 0.0
    neutral = 0.0

    for item in scored_chunks:
        chunk_id = item["chunk_id"]
        w = node_weights.get(chunk_id, 0.0)
        support += w * float(item["supports"])
        refute += w * float(item["refutes"])
        neutral += w * float(item["neutral"])

    max_sr = max(support, refute, 1e-8)
    conflict = min(support, refute) / max_sr

    return {
        "graph_support_score": support,
        "graph_refute_score": refute,
        "graph_neutral_score": neutral,
        "graph_conflict_score": conflict,
    }

def aggregate_flat_scores(
    scored_chunks: Sequence[Dict],
) -> Dict[str, float]:
    """
    Flat baseline-style summary from independently scored chunks.
    """
    if not scored_chunks:
        return {
            "flat_support_score": 0.0,
            "flat_refute_score": 0.0,
            "flat_neutral_score": 1.0,
        }

    best_support = max(float(item["supports"]) for item in scored_chunks)
    best_refute = max(float(item["refutes"]) for item in scored_chunks)
    best_neutral = max(float(item["neutral"]) for item in scored_chunks)

    return {
        "flat_support_score": best_support,
        "flat_refute_score": best_refute,
        "flat_neutral_score": best_neutral,
    }

def choose_hybrid_graph_verdict(
    scored_chunks: Sequence[Dict],
    node_weights: Dict[str, float],
    graph_alpha: float = 0.35,
    label_threshold: float = 0.40,
    margin_threshold: float = 0.05,
    conflict_threshold: float = 0.75,
    neutral_threshold: float = 0.55,
    max_evidence_chunks: int = 1,
):
    """
    Hybrid verdict:
    - graph scores improve evidence aggregation
    - flat max scores stabilize final label prediction
    """
    if not scored_chunks:
        graph_scores = {
            "graph_support_score": 0.0,
            "graph_refute_score": 0.0,
            "graph_neutral_score": 1.0,
            "graph_conflict_score": 0.0,
        }
        flat_scores = {
            "flat_support_score": 0.0,
            "flat_refute_score": 0.0,
            "flat_neutral_score": 1.0,
        }
        hybrid_scores = {
            "hybrid_support_score": 0.0,
            "hybrid_refute_score": 0.0,
            "hybrid_neutral_score": 1.0,
            "hybrid_conflict_score": 0.0,
        }
        diagnostics = {
            "graph_scores": graph_scores,
            "flat_scores": flat_scores,
            "hybrid_scores": hybrid_scores,
        }
        return "insufficient", [], diagnostics

    graph_scores = aggregate_graph_scores(scored_chunks, node_weights)
    flat_scores = aggregate_flat_scores(scored_chunks)

    support = (
        graph_alpha * graph_scores["graph_support_score"]
        + (1.0 - graph_alpha) * flat_scores["flat_support_score"]
    )
    refute = (
        graph_alpha * graph_scores["graph_refute_score"]
        + (1.0 - graph_alpha) * flat_scores["flat_refute_score"]
    )
    neutral = (
        graph_alpha * graph_scores["graph_neutral_score"]
        + (1.0 - graph_alpha) * flat_scores["flat_neutral_score"]
    )

    max_sr = max(support, refute, 1e-8)
    conflict = min(support, refute) / max_sr
    margin = abs(support - refute)
    best_sr = max(support, refute)

    hybrid_scores = {
        "hybrid_support_score": support,
        "hybrid_refute_score": refute,
        "hybrid_neutral_score": neutral,
        "hybrid_conflict_score": conflict,
    }

    diagnostics = {
        "graph_scores": graph_scores,
        "flat_scores": flat_scores,
        "hybrid_scores": hybrid_scores,
    }

    if best_sr < label_threshold:
        return "insufficient", [], diagnostics

    if conflict > conflict_threshold and neutral > neutral_threshold:
        return "insufficient", [], diagnostics

    if margin < margin_threshold and neutral > 0.40:
        return "insufficient", [], diagnostics

    if support >= refute:
        predicted_label = "supports"
        ranked = sorted(
            scored_chunks,
            key=lambda x: node_weights.get(x["chunk_id"], 0.0) * float(x["supports"]),
            reverse=True,
        )
    else:
        predicted_label = "refutes"
        ranked = sorted(
            scored_chunks,
            key=lambda x: node_weights.get(x["chunk_id"], 0.0) * float(x["refutes"]),
            reverse=True,
        )

    predicted_evidence_chunks = [item["chunk_id"] for item in ranked[:max_evidence_chunks]]
    return predicted_label, predicted_evidence_chunks, diagnostics


def build_graph_predictions(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    local_graphs_by_query_id: Dict[str, dict],
    chunks_by_id: Dict[str, Chunk],
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    batch_size: int = 8,
    max_length: int = 256,
    max_evidence_chunks: int = 1,
    top_nli_chunks: int = 3,
    graph_alpha: float = 0.35,
    label_threshold: float = 0.40,
    margin_threshold: float = 0.05,
    conflict_threshold: float = 0.75,
    neutral_threshold: float = 0.55,
) -> List[Dict]:
    query_by_id = {query.query_id: query for query in queries}

    pair_texts: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str]] = []

    total_candidate_chunks = 0
    total_scored_chunks = 0

    for graph_input in graph_inputs:
        query = query_by_id[graph_input.query_id]

        total_candidate_chunks += len(graph_input.candidate_chunks)
        selected_chunk_ids = select_top_chunks_for_nli(
            graph_input=graph_input,
            top_nli_chunks=top_nli_chunks,
        )
        total_scored_chunks += len(selected_chunk_ids)

        for chunk_id in selected_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            pair_texts.append((query.text, chunk.text))
            pair_meta.append((query.query_id, chunk_id))

    logger.info(
        f"Selected {total_scored_chunks} chunks for graph-aware NLI scoring "
        f"(from {total_candidate_chunks} retrieved chunks total)"
    )

    logger.info(f"Scoring {len(pair_texts)} claim-evidence pairs for graph-aware module")
    all_scores = score_claim_evidence_pairs(
        claim_evidence_pairs=pair_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    scored_by_query: Dict[str, List[Dict]] = {}
    for (query_id, chunk_id), scores in zip(pair_meta, all_scores):
        scored_by_query.setdefault(query_id, []).append(
            {
                "chunk_id": chunk_id,
                "supports": scores["supports"],
                "refutes": scores["refutes"],
                "neutral": scores["neutral"],
            }
        )

    predictions: List[Dict] = []

    for idx, graph_input in enumerate(graph_inputs, start=1):
        local_graph = local_graphs_by_query_id.get(graph_input.query_id, {})
        scored_chunks = scored_by_query.get(graph_input.query_id, [])

        selected_chunk_ids = [item["chunk_id"] for item in scored_chunks]
        node_weights = build_graph_node_weights(
            graph_input=graph_input,
            local_graph=local_graph,
            selected_chunk_ids=selected_chunk_ids,
        )

        predicted_label, predicted_evidence_chunks, diagnostics = choose_hybrid_graph_verdict(
            scored_chunks=scored_chunks,
            node_weights=node_weights,
            graph_alpha=graph_alpha,
            label_threshold=label_threshold,
            margin_threshold=margin_threshold,
            conflict_threshold=conflict_threshold,
            neutral_threshold=neutral_threshold,
            max_evidence_chunks=max_evidence_chunks,
        )

        top_chunks_sorted = sorted(
            scored_chunks,
            key=lambda x: max(
                float(x["supports"]),
                float(x["refutes"]),
                float(x["neutral"]),
            ),
            reverse=True,
        )[:5]

        predictions.append(
            {
                "query_id": graph_input.query_id,
                "predicted_label": predicted_label,
                "predicted_evidence_chunks": predicted_evidence_chunks,
                "graph_scores": diagnostics["graph_scores"],
                "flat_scores": diagnostics["flat_scores"],
                "hybrid_scores": diagnostics["hybrid_scores"],
                "node_weights": node_weights,
                "top_chunk_scores": top_chunks_sorted,
                "candidate_chunk_count": len(graph_input.candidate_chunks),
                "scored_chunk_count": len(scored_chunks),
            }
        )

        if idx % 100 == 0:
            logger.info(f"Built graph-aware predictions for {idx}/{len(graph_inputs)} queries")

    logger.info(f"Built {len(predictions)} graph-aware SciFact predictions")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first graph-aware SciFact verdict module.")
    parser.add_argument("--queries", required=True, help="Path to Query JSONL")
    parser.add_argument("--graph-inputs", required=True, help="Path to GraphInput JSONL")
    parser.add_argument("--local-graphs", required=True, help="Path to local graph JSONL")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")

    parser.add_argument("--output-predictions", required=True, help="Path to save predictions JSONL")
    parser.add_argument("--output-metrics", required=True, help="Path to save metrics JSON")

    parser.add_argument(
        "--model-name",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HF NLI model name",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--max-evidence-chunks", type=int, default=1)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument(
        "--graph-alpha",
        type=float,
        default=0.35,
        help="Weight assigned to graph aggregation vs flat top-chunk NLI",
    )

    parser.add_argument("--label-threshold", type=float, default=0.35)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--conflict-threshold", type=float, default=0.80)
    parser.add_argument("--neutral-threshold", type=float, default=0.55)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    graph_inputs = load_graph_inputs(args.graph_inputs)
    local_graphs = load_local_graphs(args.local_graphs)
    chunks = load_chunks(args.chunks)

    local_graphs_by_query_id = map_local_graphs_by_query_id(local_graphs)
    chunks_by_id = map_chunks_by_id(chunks)

    predictions = build_graph_predictions(
        queries=queries,
        graph_inputs=graph_inputs,
        local_graphs_by_query_id=local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_evidence_chunks=args.max_evidence_chunks,
        top_nli_chunks=args.top_nli_chunks,
        label_threshold=args.label_threshold,
        margin_threshold=args.margin_threshold,
        conflict_threshold=args.conflict_threshold,
        neutral_threshold=args.neutral_threshold,
        graph_alpha=args.graph_alpha,
    )

    write_jsonl(predictions, args.output_predictions)

    metrics = evaluate_scifact_predictions(
        queries=queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote graph-aware predictions to {args.output_predictions}")
    logger.info(f"Wrote graph-aware metrics to {args.output_metrics}")
    logger.info(f"Label macro F1: {metrics['label_metrics']['macro_f1']:.4f}")
    logger.info(f"Evidence micro F1: {metrics['evidence_metrics']['micro_f1']:.4f}")


if __name__ == "__main__":
    main()