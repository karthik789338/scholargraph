from pathlib import Path
import json
import argparse
import joblib

from src.graph.scifact_graph_feature_classifier import (
    build_feature_rows,
    train_classifier,
)
from src.graph.scifact_graph_verdict import (
    load_queries,
    load_graph_inputs,
    load_local_graphs,
    load_chunks,
    map_chunks_by_id,
    map_local_graphs_by_query_id,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--train-graph-inputs", required=True)
    parser.add_argument("--train-local-graphs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_queries = load_queries(args.train_queries)
    train_graph_inputs = load_graph_inputs(args.train_graph_inputs)
    train_local_graphs = load_local_graphs(args.train_local_graphs)

    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)
    train_local_graphs_by_query_id = map_local_graphs_by_query_id(train_local_graphs)

    X_train, y_train, feature_names, _train_rows = build_feature_rows(
        queries=train_queries,
        graph_inputs=train_graph_inputs,
        local_graphs_by_query_id=train_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_nli_chunks=args.top_nli_chunks,
    )

    model = train_classifier(X_train, y_train)

    joblib.dump(model, out_dir / "graph_feature_classifier.joblib")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    with open(out_dir / "train_only_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_train_examples": len(train_queries),
                "num_features": len(feature_names),
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "top_nli_chunks": args.top_nli_chunks,
            },
            f,
            indent=2,
        )

    print(json.dumps({
        "num_train_examples": len(train_queries),
        "num_features": len(feature_names),
        "saved_model": str(out_dir / "graph_feature_classifier.joblib"),
    }, indent=2))

if __name__ == "__main__":
    main()
