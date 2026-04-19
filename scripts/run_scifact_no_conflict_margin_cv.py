from pathlib import Path
import json
import statistics as stats

from src.graph.scifact_graph_feature_classifier import (
    build_feature_rows,
    train_classifier,
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

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
BATCH_SIZE = 32
MAX_LENGTH = 256
TOP_NLI_CHUNKS = 3
MAX_EVIDENCE_CHUNKS = 1

DROP_NAMES = {
    "graph_conflict_score",
    "graph_margin_signed",
    "graph_margin_abs",
    "best_margin_signed",
    "best_margin_abs",
}

base = Path("data/processed/cv")
fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold")])

chunks = load_chunks("data/processed/chunks/scifact_chunks.jsonl")
chunks_by_id = map_chunks_by_id(chunks)

rows = []

for fold_dir in fold_dirs:
    fold = fold_dir.name
    idx = fold.replace("fold", "")

    train_queries_path = f"data/processed/queries/scifact_fold{idx}_train_with_chunks.jsonl"
    train_graph_inputs_path = f"data/processed/graph_inputs/scifact_fold{idx}_train_graph_inputs.jsonl"
    train_local_graphs_path = f"data/processed/graph_inputs/scifact_fold{idx}_train_local_graphs.jsonl"

    dev_queries_path = f"data/processed/queries/scifact_fold{idx}_dev_with_chunks.jsonl"
    dev_graph_inputs_path = f"data/processed/graph_inputs/scifact_fold{idx}_dev_graph_inputs.jsonl"
    dev_local_graphs_path = f"data/processed/graph_inputs/scifact_fold{idx}_dev_local_graphs.jsonl"

    train_queries = load_queries(train_queries_path)
    train_graph_inputs = load_graph_inputs(train_graph_inputs_path)
    train_local_graphs = load_local_graphs(train_local_graphs_path)

    dev_queries = load_queries(dev_queries_path)
    dev_graph_inputs = load_graph_inputs(dev_graph_inputs_path)
    dev_local_graphs = load_local_graphs(dev_local_graphs_path)

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
        raise ValueError(f"Feature mismatch in {fold}")

    keep_idx = [i for i, name in enumerate(feature_names) if name not in DROP_NAMES]
    used_features = [feature_names[i] for i in keep_idx]

    X_train_sub = X_train[:, keep_idx]
    X_dev_sub = X_dev[:, keep_idx]

    model = train_classifier(X_train_sub, y_train)

    dev_predictions = build_predictions_from_classifier(
        model=model,
        X=X_dev_sub,
        rows=dev_rows,
        max_evidence_chunks=MAX_EVIDENCE_CHUNKS,
    )
    dev_metrics = evaluate_scifact_predictions(
        queries=dev_queries,
        predictions=dev_predictions,
    )

    flat_graph_summary = json.load(open(fold_dir / "summary.json", encoding="utf-8"))
    compact_summary = json.load(open(fold_dir / "graph_feature_classifier_compact" / "summary.json", encoding="utf-8"))

    row = {
        "fold": fold,
        "num_features": len(used_features),
        "dev_label_macro_f1": dev_metrics["label_metrics"]["macro_f1"],
        "dev_evidence_micro_f1": dev_metrics["evidence_metrics"]["micro_f1"],
        "flat_dev_macro_f1": flat_graph_summary["flat_dev_macro_f1"],
        "flat_dev_evidence_micro_f1": flat_graph_summary["flat_dev_evidence_micro_f1"],
        "graph_rule_dev_macro_f1": flat_graph_summary["graph_dev_macro_f1"],
        "graph_rule_dev_evidence_micro_f1": flat_graph_summary["graph_dev_evidence_micro_f1"],
        "compact_dev_macro_f1": compact_summary["dev_label_macro_f1"],
        "compact_dev_evidence_micro_f1": compact_summary["dev_evidence_micro_f1"],
    }
    rows.append(row)
    print(json.dumps(row, indent=2))

def mean_std(values):
    return {
        "mean": stats.mean(values),
        "std": stats.pstdev(values) if len(values) > 1 else 0.0,
    }

summary = {
    "folds": rows,
    "aggregate": {
        "no_conflict_margin_macro_f1": mean_std([r["dev_label_macro_f1"] for r in rows]),
        "no_conflict_margin_evidence_micro_f1": mean_std([r["dev_evidence_micro_f1"] for r in rows]),
        "flat_macro_f1": mean_std([r["flat_dev_macro_f1"] for r in rows]),
        "flat_evidence_micro_f1": mean_std([r["flat_dev_evidence_micro_f1"] for r in rows]),
        "graph_rule_macro_f1": mean_std([r["graph_rule_dev_macro_f1"] for r in rows]),
        "graph_rule_evidence_micro_f1": mean_std([r["graph_rule_dev_evidence_micro_f1"] for r in rows]),
        "compact_macro_f1": mean_std([r["compact_dev_macro_f1"] for r in rows]),
        "compact_evidence_micro_f1": mean_std([r["compact_dev_evidence_micro_f1"] for r in rows]),
        "delta_vs_compact_macro_f1": mean_std([r["dev_label_macro_f1"] - r["compact_dev_macro_f1"] for r in rows]),
        "delta_vs_compact_evidence_micro_f1": mean_std([r["dev_evidence_micro_f1"] - r["compact_dev_evidence_micro_f1"] for r in rows]),
    }
}

out_dir = Path("reports/scifact_frozen")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "scifact_no_conflict_margin_cv.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nFINAL SUMMARY")
print(json.dumps(summary, indent=2))
print(f"\nWrote {out_dir / 'scifact_no_conflict_margin_cv.json'}")
