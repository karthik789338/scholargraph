from pathlib import Path
import json
import numpy as np

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

TRAIN_QUERIES = "data/processed/queries/scifact_train_with_chunks.jsonl"
TRAIN_GRAPH_INPUTS = "data/processed/graph_inputs/scifact_train_graph_inputs.jsonl"
TRAIN_LOCAL_GRAPHS = "data/processed/graph_inputs/scifact_train_local_graphs.jsonl"

DEV_QUERIES = "data/processed/queries/scifact_dev_with_chunks.jsonl"
DEV_GRAPH_INPUTS = "data/processed/graph_inputs/scifact_dev_graph_inputs.jsonl"
DEV_LOCAL_GRAPHS = "data/processed/graph_inputs/scifact_dev_local_graphs.jsonl"

CHUNKS = "data/processed/chunks/scifact_chunks.jsonl"

OUT_DIR = Path("reports/scifact_frozen/ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
BATCH_SIZE = 32
MAX_LENGTH = 256
TOP_NLI_CHUNKS = 3
MAX_EVIDENCE_CHUNKS = 1


def subset_matrix(X, feature_names, drop_names):
    keep_idx = [i for i, name in enumerate(feature_names) if name not in drop_names]
    keep_names = [feature_names[i] for i in keep_idx]
    X2 = X[:, keep_idx]
    return X2, keep_names


def run_variant(name, X_train, X_dev, y_train, train_rows, dev_rows, feature_names, drop_names):
    Xtr, used_names = subset_matrix(X_train, feature_names, drop_names)
    Xdv, _ = subset_matrix(X_dev, feature_names, drop_names)

    model = train_classifier(Xtr, y_train)

    train_predictions = build_predictions_from_classifier(
        model=model,
        X=Xtr,
        rows=train_rows,
        max_evidence_chunks=MAX_EVIDENCE_CHUNKS,
    )
    dev_predictions = build_predictions_from_classifier(
        model=model,
        X=Xdv,
        rows=dev_rows,
        max_evidence_chunks=MAX_EVIDENCE_CHUNKS,
    )

    train_metrics = evaluate_scifact_predictions(train_queries, train_predictions)
    dev_metrics = evaluate_scifact_predictions(dev_queries, dev_predictions)

    return {
        "variant": name,
        "num_features": len(used_names),
        "features_used": used_names,
        "train_label_macro_f1": train_metrics["label_metrics"]["macro_f1"],
        "train_evidence_micro_f1": train_metrics["evidence_metrics"]["micro_f1"],
        "dev_label_macro_f1": dev_metrics["label_metrics"]["macro_f1"],
        "dev_evidence_micro_f1": dev_metrics["evidence_metrics"]["micro_f1"],
    }


train_queries = load_queries(TRAIN_QUERIES)
train_graph_inputs = load_graph_inputs(TRAIN_GRAPH_INPUTS)
train_local_graphs = load_local_graphs(TRAIN_LOCAL_GRAPHS)

dev_queries = load_queries(DEV_QUERIES)
dev_graph_inputs = load_graph_inputs(DEV_GRAPH_INPUTS)
dev_local_graphs = load_local_graphs(DEV_LOCAL_GRAPHS)

chunks = load_chunks(CHUNKS)
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

assert feature_names == dev_feature_names

feature_groups = {
    "retrieval_strength": {
        "max_retrieval_score",
        "mean_retrieval_score",
    },
    "graph_statistics": {
        "node_weight_entropy",
        "max_node_weight",
        "min_node_weight",
        "mean_node_weight",
        "num_scored_chunks",
    },
    "conflict_margin": {
        "graph_conflict_score",
        "graph_margin_signed",
        "graph_margin_abs",
        "best_margin_signed",
        "best_margin_abs",
    },
}

variants = [
    ("compact_full", set()),
    ("no_retrieval_strength", feature_groups["retrieval_strength"]),
    ("no_graph_statistics", feature_groups["graph_statistics"]),
    ("no_conflict_margin", feature_groups["conflict_margin"]),
    (
        "only_retrieval_and_best_local",
        feature_groups["graph_statistics"] | feature_groups["conflict_margin"],
    ),
]

results = []
for name, drop_names in variants:
    row = run_variant(
        name=name,
        X_train=X_train,
        X_dev=X_dev,
        y_train=y_train,
        train_rows=train_rows,
        dev_rows=dev_rows,
        feature_names=feature_names,
        drop_names=drop_names,
    )
    results.append(row)
    print(json.dumps(row, indent=2))

flat = json.load(open("data/processed/baselines/scifact_dev_baseline_metrics_top3.json", encoding="utf-8"))
graph_rule = json.load(open("data/processed/dev_sweeps/scifact_dev_graph_best_thresholds.json", encoding="utf-8"))

summary = {
    "flat_baseline_dev": {
        "label_macro_f1": flat["label_metrics"]["macro_f1"],
        "evidence_micro_f1": flat["evidence_metrics"]["micro_f1"],
    },
    "graph_rule_dev": {
        "label_macro_f1": graph_rule["label_macro_f1"],
        "evidence_micro_f1": graph_rule["evidence_micro_f1"],
    },
    "compact_ablation": results,
}

with open(OUT_DIR / "scifact_feature_ablation.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

md = []
md.append("# SciFact Feature Ablation")
md.append("")
md.append("| Model | #Features | Dev Macro F1 | Dev Evidence Micro F1 |")
md.append("|---|---:|---:|---:|")
md.append(f"| Flat baseline | -- | {flat['label_metrics']['macro_f1']:.4f} | {flat['evidence_metrics']['micro_f1']:.4f} |")
md.append(f"| Graph rule | -- | {graph_rule['label_macro_f1']:.4f} | {graph_rule['evidence_micro_f1']:.4f} |")
for row in results:
    md.append(
        f"| {row['variant']} | {row['num_features']} | "
        f"{row['dev_label_macro_f1']:.4f} | {row['dev_evidence_micro_f1']:.4f} |"
    )

with open(OUT_DIR / "scifact_feature_ablation.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"\nWrote {OUT_DIR/'scifact_feature_ablation.json'}")
print(f"Wrote {OUT_DIR/'scifact_feature_ablation.md'}")
