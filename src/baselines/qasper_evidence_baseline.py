from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from src.eval.qasper_metrics import compute_evidence_metrics
from src.graph.schemas import GraphInput, Query
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_graph_inputs(path: str | Path) -> List[GraphInput]:
    return [GraphInput(**x) for x in read_jsonl(path)]


def build_predictions(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    top_k: int = 3,
) -> List[Dict]:
    predictions = []

    query_map = {q.query_id: q for q in queries}

    for graph_input in graph_inputs:
        retrieval_scores = graph_input.metadata.get("retrieval_scores", {})

        ranked = sorted(
            graph_input.candidate_chunks,
            key=lambda cid: (
                retrieval_scores.get(cid, {}).get("rank", 10**9),
                -float(retrieval_scores.get(cid, {}).get("score", 0.0)),
            ),
        )

        pred_chunks = ranked[:top_k]

        predictions.append(
            {
                "query_id": graph_input.query_id,
                "predicted_evidence_chunks": pred_chunks,
                "candidate_chunk_count": len(graph_input.candidate_chunks),
                "scored_chunk_count": min(top_k, len(graph_input.candidate_chunks)),
            }
        )

    logger.info(f"Built {len(predictions)} QASPER evidence predictions")
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
    parser = argparse.ArgumentParser(description="Run QASPER evidence retrieval baseline")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--graph-inputs", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    graph_inputs = load_graph_inputs(args.graph_inputs)

    predictions = build_predictions(
        queries=queries,
        graph_inputs=graph_inputs,
        top_k=args.top_k,
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