from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.scifact_graph_verdict import choose_hybrid_graph_verdict
from src.graph.schemas import Query
from src.utils.io import read_jsonl, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_predictions(path: str | Path) -> List[dict]:
    return read_jsonl(path)


def rebuild_predictions_with_thresholds(
    saved_predictions: List[dict],
    graph_alpha: float,
    label_threshold: float,
    margin_threshold: float,
    conflict_threshold: float,
    neutral_threshold: float,
    max_evidence_chunks: int = 1,
) -> List[dict]:
    rebuilt = []

    for pred in saved_predictions:
        scored_chunks = pred.get("top_chunk_scores", [])
        node_weights = pred.get("node_weights", {})

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

        rebuilt.append(
            {
                "query_id": pred["query_id"],
                "predicted_label": predicted_label,
                "predicted_evidence_chunks": predicted_evidence_chunks,
                "graph_scores": diagnostics["graph_scores"],
                "flat_scores": diagnostics["flat_scores"],
                "hybrid_scores": diagnostics["hybrid_scores"],
                "node_weights": node_weights,
                "top_chunk_scores": scored_chunks,
            }
        )

    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hybrid graph verdict thresholds offline.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--predictions", required=True, help="Saved graph prediction JSONL")
    parser.add_argument("--output", required=True, help="Path to save best settings JSON")

    args = parser.parse_args()

    queries = load_queries(args.queries)
    saved_predictions = load_predictions(args.predictions)

    graph_alpha_grid = [0.20, 0.25, 0.30, 0.35, 0.40]
    label_threshold_grid = [0.35, 0.40, 0.45, 0.50]
    margin_threshold_grid = [0.03, 0.05, 0.07]
    conflict_threshold_grid = [0.70, 0.75, 0.80, 0.85]
    neutral_threshold_grid = [0.50, 0.55, 0.60, 0.65]

    best = None
    all_results = []

    total = (
        len(graph_alpha_grid)
        * len(label_threshold_grid)
        * len(margin_threshold_grid)
        * len(conflict_threshold_grid)
        * len(neutral_threshold_grid)
    )
    seen = 0

    for graph_alpha, label_threshold, margin_threshold, conflict_threshold, neutral_threshold in itertools.product(
        graph_alpha_grid,
        label_threshold_grid,
        margin_threshold_grid,
        conflict_threshold_grid,
        neutral_threshold_grid,
    ):
        rebuilt = rebuild_predictions_with_thresholds(
            saved_predictions=saved_predictions,
            graph_alpha=graph_alpha,
            label_threshold=label_threshold,
            margin_threshold=margin_threshold,
            conflict_threshold=conflict_threshold,
            neutral_threshold=neutral_threshold,
            max_evidence_chunks=1,
        )

        metrics = evaluate_scifact_predictions(queries=queries, predictions=rebuilt)

        row = {
            "graph_alpha": graph_alpha,
            "label_threshold": label_threshold,
            "margin_threshold": margin_threshold,
            "conflict_threshold": conflict_threshold,
            "neutral_threshold": neutral_threshold,
            "label_macro_f1": metrics["label_metrics"]["macro_f1"],
            "evidence_micro_f1": metrics["evidence_metrics"]["micro_f1"],
            "accuracy": metrics["label_metrics"]["accuracy"],
        }
        all_results.append(row)

        if best is None:
            best = row
        else:
            if (
                row["label_macro_f1"] > best["label_macro_f1"]
                or (
                    row["label_macro_f1"] == best["label_macro_f1"]
                    and row["evidence_micro_f1"] > best["evidence_micro_f1"]
                )
            ):
                best = row

        seen += 1
        if seen % 100 == 0 or seen == total:
            logger.info(f"Swept {seen}/{total} settings")

    all_results = sorted(
        all_results,
        key=lambda x: (x["label_macro_f1"], x["evidence_micro_f1"], x["accuracy"]),
        reverse=True,
    )

    output = {
        "best": best,
        "top10": all_results[:10],
        "total_settings": total,
    }

    write_json(output, args.output)

    logger.info(f"Best setting: {best}")
    logger.info(f"Wrote sweep results to {args.output}")


if __name__ == "__main__":
    main()