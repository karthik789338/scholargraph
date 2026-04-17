from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from src.baselines.qasper_answer_baseline import (
    build_predictions,
    load_chunks,
    load_queries,
    map_chunks_by_id,
)
from src.eval.qasper_answer_metrics import evaluate_qasper_answer_predictions
from src.graph.schemas import Query
from src.utils.io import write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_oracle_evidence_predictions(
    queries: Sequence[Query],
    max_gold_chunks: int | None = None,
) -> List[Dict]:
    predictions: List[Dict] = []

    for query in queries:
        seen = set()
        gold_chunk_ids: List[str] = []

        for ev in query.gold_evidence:
            if ev.chunk_id is not None and ev.chunk_id not in seen:
                seen.add(ev.chunk_id)
                gold_chunk_ids.append(ev.chunk_id)

        if max_gold_chunks is not None:
            gold_chunk_ids = gold_chunk_ids[:max_gold_chunks]

        predictions.append(
            {
                "query_id": query.query_id,
                "predicted_evidence_chunks": gold_chunk_ids,
                "oracle": True,
                "gold_chunk_count": len(gold_chunk_ids),
            }
        )

    logger.info(f"Built {len(predictions)} oracle-evidence predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run QASPER oracle answer baseline")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--max-gold-chunks",
        type=int,
        default=None,
        help="Optional cap on number of gold evidence chunks per query",
    )

    args = parser.parse_args()

    queries = load_queries(args.queries)
    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)

    oracle_evidence_predictions = build_oracle_evidence_predictions(
        queries=queries,
        max_gold_chunks=args.max_gold_chunks,
    )

    predictions = build_predictions(
        queries=queries,
        evidence_predictions=oracle_evidence_predictions,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(predictions, args.output_predictions)

    metrics = evaluate_qasper_answer_predictions(
        queries=queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote predictions to {args.output_predictions}")
    logger.info(f"Wrote metrics to {args.output_metrics}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Token F1: {metrics['token_f1']:.4f}")


if __name__ == "__main__":
    main()