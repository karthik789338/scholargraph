from pathlib import Path
import sys
import json
import itertools
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.scifact_graph_verdict import (
    load_queries,
    load_graph_inputs,
    load_chunks,
    load_local_graphs,
    map_chunks_by_id,
    map_local_graphs_by_query_id,
    build_graph_predictions,
)
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.utils.io import ensure_dir, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep graph verdict thresholds on SciFact dev.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--graph-inputs", required=True)
    parser.add_argument("--local-graphs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)

    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    queries = load_queries(args.queries)
    graph_inputs = load_graph_inputs(args.graph_inputs)
    local_graphs = load_local_graphs(args.local_graphs)
    chunks = load_chunks(args.chunks)

    local_graphs_by_query_id = map_local_graphs_by_query_id(local_graphs)
    chunks_by_id = map_chunks_by_id(chunks)

    label_thresholds = [0.30, 0.35, 0.40, 0.45]
    margin_thresholds = [0.03, 0.05, 0.08]
    conflict_thresholds = [0.65, 0.75, 0.85]
    neutral_thresholds = [0.45, 0.55, 0.65]

    results = []
    best = None

    total = (
        len(label_thresholds)
        * len(margin_thresholds)
        * len(conflict_thresholds)
        * len(neutral_thresholds)
    )
    run_idx = 0

    for label_threshold, margin_threshold, conflict_threshold, neutral_threshold in itertools.product(
        label_thresholds,
        margin_thresholds,
        conflict_thresholds,
        neutral_thresholds,
    ):
        run_idx += 1
        logger.info(
            f"Sweep run {run_idx}/{total}: "
            f"label={label_threshold}, margin={margin_threshold}, "
            f"conflict={conflict_threshold}, neutral={neutral_threshold}"
        )

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
            label_threshold=label_threshold,
            margin_threshold=margin_threshold,
            conflict_threshold=conflict_threshold,
            neutral_threshold=neutral_threshold,
        )

        metrics = evaluate_scifact_predictions(
            queries=queries,
            predictions=predictions,
        )

        label_macro_f1 = metrics["label_metrics"]["macro_f1"]
        evidence_micro_f1 = metrics["evidence_metrics"]["micro_f1"]

        row = {
            "label_threshold": label_threshold,
            "margin_threshold": margin_threshold,
            "conflict_threshold": conflict_threshold,
            "neutral_threshold": neutral_threshold,
            "label_macro_f1": label_macro_f1,
            "evidence_micro_f1": evidence_micro_f1,
            "metrics": metrics,
        }
        results.append(row)

        if best is None:
            best = row
        else:
            # Primary objective: macro F1
            # Secondary objective: evidence micro F1
            if (
                row["label_macro_f1"] > best["label_macro_f1"]
                or (
                    row["label_macro_f1"] == best["label_macro_f1"]
                    and row["evidence_micro_f1"] > best["evidence_micro_f1"]
                )
            ):
                best = row

    results_sorted = sorted(
        results,
        key=lambda x: (x["label_macro_f1"], x["evidence_micro_f1"]),
        reverse=True,
    )

    write_json(results_sorted, output_dir / "scifact_dev_graph_threshold_sweep.json")
    write_json(best, output_dir / "scifact_dev_graph_best_thresholds.json")

    logger.info("Top 5 threshold settings:")
    for row in results_sorted[:5]:
        logger.info(
            f"macro_f1={row['label_macro_f1']:.4f}, "
            f"evidence_f1={row['evidence_micro_f1']:.4f}, "
            f"label={row['label_threshold']}, "
            f"margin={row['margin_threshold']}, "
            f"conflict={row['conflict_threshold']}, "
            f"neutral={row['neutral_threshold']}"
        )

    logger.info(f"Best dev macro F1: {best['label_macro_f1']:.4f}")
    logger.info(f"Best dev evidence micro F1: {best['evidence_micro_f1']:.4f}")


if __name__ == "__main__":
    main()
