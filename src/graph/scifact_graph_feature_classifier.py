from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.scifact_baseline import (
    score_claim_evidence_pairs,
    select_top_chunks_for_nli,
)
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.schemas import Chunk, GraphInput, Query
from src.graph.scifact_graph_verdict import (
    aggregate_graph_scores,
    build_graph_node_weights,
    extract_evidence_degree_weights,
    load_chunks,
    load_graph_inputs,
    load_local_graphs,
    load_queries,
    map_chunks_by_id,
    map_local_graphs_by_query_id,
)
from src.utils.io import ensure_dir, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

LABEL_TO_ID = {
    "supports": 0,
    "refutes": 1,
    "insufficient": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def score_selected_chunks(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    chunks_by_id: Dict[str, Chunk],
    model_name: str,
    batch_size: int,
    max_length: int,
    top_nli_chunks: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
    query_by_id = {query.query_id: query for query in queries}

    pair_texts: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str]] = []
    selected_chunk_ids_by_query: Dict[str, List[str]] = {}

    total_candidate_chunks = 0
    total_scored_chunks = 0

    for graph_input in graph_inputs:
        query = query_by_id[graph_input.query_id]
        total_candidate_chunks += len(graph_input.candidate_chunks)

        selected_chunk_ids = select_top_chunks_for_nli(
            graph_input=graph_input,
            top_nli_chunks=top_nli_chunks,
        )
        selected_chunk_ids_by_query[graph_input.query_id] = selected_chunk_ids
        total_scored_chunks += len(selected_chunk_ids)

        for chunk_id in selected_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            pair_texts.append((query.text, chunk.text))
            pair_meta.append((query.query_id, chunk_id))

    logger.info(
        f"Selected {total_scored_chunks} chunks for graph-feature NLI scoring "
        f"(from {total_candidate_chunks} retrieved chunks total)"
    )
    logger.info(f"Scoring {len(pair_texts)} claim-evidence pairs for graph-feature classifier")

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

    return scored_by_query, selected_chunk_ids_by_query


def build_feature_dict(
    graph_input: GraphInput,
    local_graph: dict,
    scored_chunks: Sequence[Dict],
    selected_chunk_ids: Sequence[str],
    top_nli_chunks: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    retrieval_scores = graph_input.metadata.get("retrieval_scores", {})
    node_weights = build_graph_node_weights(
        graph_input=graph_input,
        local_graph=local_graph,
        selected_chunk_ids=selected_chunk_ids,
    )

    agg = aggregate_graph_scores(
        scored_chunks=scored_chunks,
        node_weights=node_weights,
    )

    if scored_chunks:
        best_support = max(float(item["supports"]) for item in scored_chunks)
        best_refute = max(float(item["refutes"]) for item in scored_chunks)
        best_neutral = max(float(item["neutral"]) for item in scored_chunks)

        support_gt_05 = sum(1 for item in scored_chunks if float(item["supports"]) > 0.5)
        refute_gt_05 = sum(1 for item in scored_chunks if float(item["refutes"]) > 0.5)
        neutral_gt_05 = sum(1 for item in scored_chunks if float(item["neutral"]) > 0.5)
        support_gt_07 = sum(1 for item in scored_chunks if float(item["supports"]) > 0.7)
        refute_gt_07 = sum(1 for item in scored_chunks if float(item["refutes"]) > 0.7)
    else:
        best_support = 0.0
        best_refute = 0.0
        best_neutral = 1.0
        support_gt_05 = 0
        refute_gt_05 = 0
        neutral_gt_05 = 0
        support_gt_07 = 0
        refute_gt_07 = 0

    weights = list(node_weights.values())
    if weights:
        eps = 1e-12
        entropy = -sum(w * np.log(max(w, eps)) for w in weights)
        max_node_weight = max(weights)
        min_node_weight = min(weights)
        mean_node_weight = float(sum(weights) / len(weights))
    else:
        entropy = 0.0
        max_node_weight = 0.0
        min_node_weight = 0.0
        mean_node_weight = 0.0

    retrieval_values = []
    for chunk_id in selected_chunk_ids:
        info = retrieval_scores.get(chunk_id, {})
        retrieval_values.append(float(info.get("score", 0.0)))

    max_retrieval_score = max(retrieval_values) if retrieval_values else 0.0
    mean_retrieval_score = float(sum(retrieval_values) / len(retrieval_values)) if retrieval_values else 0.0

    features: Dict[str, float] = {
        # graph aggregate features
        "graph_support_score": float(agg["graph_support_score"]),
        "graph_refute_score": float(agg["graph_refute_score"]),
        "graph_neutral_score": float(agg["graph_neutral_score"]),
        "graph_conflict_score": float(agg["graph_conflict_score"]),
        "graph_margin_signed": float(agg["graph_support_score"] - agg["graph_refute_score"]),
        "graph_margin_abs": float(abs(agg["graph_support_score"] - agg["graph_refute_score"])),

        # strongest local signals
        "best_support_score": float(best_support),
        "best_refute_score": float(best_refute),
        "best_neutral_score": float(best_neutral),
        "best_margin_signed": float(best_support - best_refute),
        "best_margin_abs": float(abs(best_support - best_refute)),

        # evidence count features
        "support_gt_05": float(support_gt_05),
        "refute_gt_05": float(refute_gt_05),
        "neutral_gt_05": float(neutral_gt_05),
        "support_gt_07": float(support_gt_07),
        "refute_gt_07": float(refute_gt_07),

        # node-weight distribution features
        "node_weight_entropy": float(entropy),
        "max_node_weight": float(max_node_weight),
        "min_node_weight": float(min_node_weight),
        "mean_node_weight": float(mean_node_weight),

        # compact retrieval strength features
        "max_retrieval_score": float(max_retrieval_score),
        "mean_retrieval_score": float(mean_retrieval_score),

        # minimal size signal
        "num_scored_chunks": float(len(scored_chunks)),
    }

    return features, node_weights, agg
def build_feature_rows(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    local_graphs_by_query_id: Dict[str, dict],
    chunks_by_id: Dict[str, Chunk],
    model_name: str,
    batch_size: int,
    max_length: int,
    top_nli_chunks: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    query_by_id = {query.query_id: query for query in queries}

    scored_by_query, selected_chunk_ids_by_query = score_selected_chunks(
        queries=queries,
        graph_inputs=graph_inputs,
        chunks_by_id=chunks_by_id,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        top_nli_chunks=top_nli_chunks,
    )

    rows: List[Dict] = []

    for graph_input in graph_inputs:
        query = query_by_id[graph_input.query_id]
        local_graph = local_graphs_by_query_id.get(graph_input.query_id, {})
        scored_chunks = scored_by_query.get(graph_input.query_id, [])
        selected_chunk_ids = selected_chunk_ids_by_query.get(graph_input.query_id, [])

        features, node_weights, graph_scores = build_feature_dict(
            graph_input=graph_input,
            local_graph=local_graph,
            scored_chunks=scored_chunks,
            selected_chunk_ids=selected_chunk_ids,
            top_nli_chunks=top_nli_chunks,
        )

        rows.append(
            {
                "query_id": graph_input.query_id,
                "gold_label": query.gold_label,
                "features": features,
                "scored_chunks": scored_chunks,
                "node_weights": node_weights,
                "graph_scores": graph_scores,
            }
        )

    if not rows:
        raise ValueError("No feature rows were built.")

    feature_names = list(rows[0]["features"].keys())
    X = np.asarray(
        [[row["features"][name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([LABEL_TO_ID[row["gold_label"]] for row in rows], dtype=np.int64)

    return X, y, feature_names, rows


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=0.3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def rank_evidence_chunks(
    predicted_label: str,
    scored_chunks: Sequence[Dict],
    node_weights: Dict[str, float],
    max_evidence_chunks: int,
) -> List[str]:
    if predicted_label == "insufficient":
        return []

    if predicted_label == "supports":
        ranked = sorted(
            scored_chunks,
            key=lambda x: node_weights.get(x["chunk_id"], 0.0) * float(x["supports"]),
            reverse=True,
        )
    else:
        ranked = sorted(
            scored_chunks,
            key=lambda x: node_weights.get(x["chunk_id"], 0.0) * float(x["refutes"]),
            reverse=True,
        )

    return [item["chunk_id"] for item in ranked[:max_evidence_chunks]]


def build_predictions_from_classifier(
    model: Pipeline,
    X: np.ndarray,
    rows: Sequence[Dict],
    max_evidence_chunks: int,
) -> List[Dict]:
    prob_matrix = model.predict_proba(X)
    pred_ids = model.predict(X)
    classes = list(model.named_steps["clf"].classes_)

    predictions: List[Dict] = []
    for row, pred_id, probs in zip(rows, pred_ids, prob_matrix):
        predicted_label = ID_TO_LABEL[int(pred_id)]
        prob_dict = {
            ID_TO_LABEL[int(class_id)]: float(prob)
            for class_id, prob in zip(classes, probs)
        }

        predicted_evidence_chunks = rank_evidence_chunks(
            predicted_label=predicted_label,
            scored_chunks=row["scored_chunks"],
            node_weights=row["node_weights"],
            max_evidence_chunks=max_evidence_chunks,
        )

        predictions.append(
            {
                "query_id": row["query_id"],
                "predicted_label": predicted_label,
                "predicted_evidence_chunks": predicted_evidence_chunks,
                "classifier_probs": prob_dict,
                "graph_scores": row["graph_scores"],
                "node_weights": row["node_weights"],
                "top_chunk_scores": row["scored_chunks"][:5],
            }
        )

    return predictions


def summarize_feature_importance(
    model: Pipeline,
    feature_names: Sequence[str],
    top_k: int = 12,
) -> Dict:
    clf = model.named_steps["clf"]
    coef = clf.coef_
    classes = clf.classes_

    summary = {}
    for class_id, weights in zip(classes, coef):
        label = ID_TO_LABEL[int(class_id)]
        ranked = sorted(
            zip(feature_names, weights),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )[:top_k]
        summary[label] = [
            {"feature": name, "weight": float(weight)}
            for name, weight in ranked
        ]
    return summary
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a graph-feature classifier for SciFact verdict prediction.")
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--train-graph-inputs", required=True)
    parser.add_argument("--train-local-graphs", required=True)
    parser.add_argument("--dev-queries", required=True)
    parser.add_argument("--dev-graph-inputs", required=True)
    parser.add_argument("--dev-local-graphs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--model-name",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HF NLI model name",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)

    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    train_queries = load_queries(args.train_queries)
    train_graph_inputs = load_graph_inputs(args.train_graph_inputs)
    train_local_graphs = load_local_graphs(args.train_local_graphs)

    dev_queries = load_queries(args.dev_queries)
    dev_graph_inputs = load_graph_inputs(args.dev_graph_inputs)
    dev_local_graphs = load_local_graphs(args.dev_local_graphs)

    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)

    train_local_graphs_by_query_id = map_local_graphs_by_query_id(train_local_graphs)
    dev_local_graphs_by_query_id = map_local_graphs_by_query_id(dev_local_graphs)

    X_train, y_train, feature_names, train_rows = build_feature_rows(
        queries=train_queries,
        graph_inputs=train_graph_inputs,
        local_graphs_by_query_id=train_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_nli_chunks=args.top_nli_chunks,
    )

    X_dev, y_dev, dev_feature_names, dev_rows = build_feature_rows(
        queries=dev_queries,
        graph_inputs=dev_graph_inputs,
        local_graphs_by_query_id=dev_local_graphs_by_query_id,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        top_nli_chunks=args.top_nli_chunks,
    )

    if feature_names != dev_feature_names:
        raise ValueError("Train/dev feature names do not match.")

    logger.info(f"Training graph-feature classifier on {len(X_train)} examples with {len(feature_names)} features")
    model = train_classifier(X_train, y_train)

    train_predictions = build_predictions_from_classifier(
        model=model,
        X=X_train,
        rows=train_rows,
        max_evidence_chunks=args.max_evidence_chunks,
    )
    dev_predictions = build_predictions_from_classifier(
        model=model,
        X=X_dev,
        rows=dev_rows,
        max_evidence_chunks=args.max_evidence_chunks,
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
    write_json(
        {
            "feature_names": feature_names,
            "train_num_examples": len(X_train),
            "dev_num_examples": len(X_dev),
            "train_label_macro_f1": train_metrics["label_metrics"]["macro_f1"],
            "train_evidence_micro_f1": train_metrics["evidence_metrics"]["micro_f1"],
            "dev_label_macro_f1": dev_metrics["label_metrics"]["macro_f1"],
            "dev_evidence_micro_f1": dev_metrics["evidence_metrics"]["micro_f1"],
        },
        output_dir / "summary.json",
    )

    logger.info(f"Train label macro F1: {train_metrics['label_metrics']['macro_f1']:.4f}")
    logger.info(f"Train evidence micro F1: {train_metrics['evidence_metrics']['micro_f1']:.4f}")
    logger.info(f"Dev label macro F1: {dev_metrics['label_metrics']['macro_f1']:.4f}")
    logger.info(f"Dev evidence micro F1: {dev_metrics['evidence_metrics']['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
