from pathlib import Path
import json
import argparse
import joblib

from src.graph.scifact_graph_feature_classifier import (
    build_feature_rows,
    build_predictions_from_classifier,
)
from src.graph.scifact_graph_verdict import (
    load_queries,
    load_graph_inputs,
    load_local_graphs,
    load_chunks,
    map_chunks_by_id,
    map_local_graphs_by_query_id,
)
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.utils.io import write_json, write_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dev-queries", required=True)
    parser.add_argument("--dev-graph-inputs", required=True)
    parser.add_argument("--dev-local-graphs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "graph_feature_classifier.joblib")
    feature_names_saved = json.load(open(model_dir / "feature_names.json", encoding="utf-8"))

    dev_queries = load_queries(args.dev_queries)
    dev_graph_inputs = load_graph_inputs(args.dev_graph_inputs)
    dev_local_graphs = load_local_graphs(args.dev_local_graphs)

    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)
    dev_local_graphs_by_query_id = map_local_graphs_by_query_id(dev_local_graphs)

    X_dev, y_dev, feature_names, dev_rows = build_feature_rows(
        queries=dev_queries,
        graph_inputs=dev_graph_inputs,
        local_graphs_by_query_id=dev_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_nli_chunks=args.top_nli_chunks,
    )

    if feature_names != feature_names_saved:
        raise ValueError("Feature name mismatch between saved model and dev inference features.")

    predictions = build_predictions_from_classifier(
        model=model,
        X=X_dev,
        rows=dev_rows,
        max_evidence_chunks=args.max_evidence_chunks,
    )
    metrics = evaluate_scifact_predictions(
        queries=dev_queries,
        predictions=predictions,
    )

    write_jsonl(predictions, args.output_predictions)
    write_json(metrics, args.output_metrics)

    print(json.dumps({
        "num_dev_examples": len(dev_queries),
        "label_macro_f1": metrics["label_metrics"]["macro_f1"],
        "evidence_micro_f1": metrics["evidence_metrics"]["micro_f1"],
    }, indent=2))

if __name__ == "__main__":
    main()
