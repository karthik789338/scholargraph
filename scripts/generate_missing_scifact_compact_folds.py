from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.graph.scifact_graph_feature_classifier import (
    build_feature_rows,
    train_classifier,
    build_predictions_from_classifier,
    summarize_feature_importance,
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
from src.utils.io import ensure_dir, write_json, write_jsonl

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
BATCH_SIZE = 32
MAX_LENGTH = 256
TOP_NLI_CHUNKS = 3
MAX_EVIDENCE_CHUNKS = 1


def run_fold(fold_idx: int, chunks_path: str, cv_root: str) -> None:
    fold_name = f"fold{fold_idx}"
    fold_dir = Path(cv_root) / fold_name
    output_dir = ensure_dir(fold_dir / "graph_feature_classifier_compact")

    train_queries_path = f"data/processed/queries/scifact_fold{fold_idx}_train_with_chunks.jsonl"
    train_graph_inputs_path = f"data/processed/graph_inputs/scifact_fold{fold_idx}_train_graph_inputs.jsonl"
    train_local_graphs_path = f"data/processed/graph_inputs/scifact_fold{fold_idx}_train_local_graphs.jsonl"

    dev_queries_path = f"data/processed/queries/scifact_fold{fold_idx}_dev_with_chunks.jsonl"
    dev_graph_inputs_path = f"data/processed/graph_inputs/scifact_fold{fold_idx}_dev_graph_inputs.jsonl"
    dev_local_graphs_path = f"data/processed/graph_inputs/scifact_fold{fold_idx}_dev_local_graphs.jsonl"

    print(f"\n=== {fold_name} ===")
    print(train_queries_path)
    print(train_graph_inputs_path)
    print(train_local_graphs_path)
    print(dev_queries_path)
    print(dev_graph_inputs_path)
    print(dev_local_graphs_path)

    train_queries = load_queries(train_queries_path)
    train_graph_inputs = load_graph_inputs(train_graph_inputs_path)
    train_local_graphs = load_local_graphs(train_local_graphs_path)

    dev_queries = load_queries(dev_queries_path)
    dev_graph_inputs = load_graph_inputs(dev_graph_inputs_path)
    dev_local_graphs = load_local_graphs(dev_local_graphs_path)

    chunks = load_chunks(chunks_path)
    chunks_by_id = map_chunks_by_id(chunks)

    train_local_graphs_by_query_id = map_local_graphs_by_query_id(train_local_graphs)
    dev_local_graphs_by_query_id = map_local_graphs_by_query_id(dev_local_graphs)

    X_train, y_train, feature_names, train_rows = build_feature_rows(
        queries=train_queries,
        graph_inputs=train_graph_inputs,
        local_graphs_by_query_id=train_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        top_nli_chunks=TOP_NLI_CHUNKS,
    )

    X_dev, y_dev, dev_feature_names, dev_rows = build_feature_rows(
        queries=dev_queries,
        graph_inputs=dev_graph_inputs,
        local_graphs_by_query_id=dev_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        top_nli_chunks=TOP_NLI_CHUNKS,
    )

    if feature_names != dev_feature_names:
        raise ValueError(f"Feature mismatch in {fold_name}")

    model = train_classifier(X_train, y_train)

    train_predictions = build_predictions_from_classifier(
        model=model,
        X=X_train,
        rows=train_rows,
        max_evidence_chunks=MAX_EVIDENCE_CHUNKS,
    )
    dev_predictions = build_predictions_from_classifier(
        model=model,
        X=X_dev,
        rows=dev_rows,
        max_evidence_chunks=MAX_EVIDENCE_CHUNKS,
    )

    train_metrics = evaluate_scifact_predictions(
        queries=train_queries,
        predictions=train_predictions,
    )
    dev_metrics = evaluate_scifact_predictions(
        queries=dev_queries,
        predictions=dev_predictions,
    )

    feature_importance = summarize_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    joblib.dump(model, output_dir / "graph_feature_classifier.joblib")
    write_jsonl(train_predictions, output_dir / "train_predictions.jsonl")
    write_jsonl(dev_predictions, output_dir / "dev_predictions.jsonl")
    write_json(train_metrics, output_dir / "train_metrics.json")
    write_json(dev_metrics, output_dir / "dev_metrics.json")
    write_json(feature_importance, output_dir / "feature_importance.json")

    summary = {
        "feature_names": feature_names,
        "train_num_examples": len(X_train),
        "dev_num_examples": len(X_dev),
        "train_label_macro_f1": train_metrics["label_metrics"]["macro_f1"],
        "train_evidence_micro_f1": train_metrics["evidence_metrics"]["micro_f1"],
        "dev_label_macro_f1": dev_metrics["label_metrics"]["macro_f1"],
        "dev_evidence_micro_f1": dev_metrics["evidence_metrics"]["micro_f1"],
    }
    write_json(summary, output_dir / "summary.json")

    print(json.dumps({
        "fold": fold_name,
        "dev_label_macro_f1": summary["dev_label_macro_f1"],
        "dev_evidence_micro_f1": summary["dev_evidence_micro_f1"],
        "num_features": len(feature_names),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--chunks", default="data/processed/chunks/scifact_chunks.jsonl")
    parser.add_argument("--cv-root", default="data/processed/cv")
    args = parser.parse_args()

    for fold_idx in args.folds:
        run_fold(fold_idx, args.chunks, args.cv_root)


if __name__ == "__main__":
    main()
