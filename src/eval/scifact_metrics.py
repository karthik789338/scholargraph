from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from src.graph.schemas import Query


LABELS = ["supports", "refutes", "insufficient"]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_label_metrics(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
) -> Dict:
    assert len(gold_labels) == len(pred_labels), "gold/pred lengths must match"

    per_label = {}
    total_correct = 0

    for label in LABELS:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p != label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = f1_score(precision, recall)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_correct = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p)
    accuracy = safe_div(total_correct, len(gold_labels))

    macro_precision = sum(per_label[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_label[label]["recall"] for label in LABELS) / len(LABELS)
    macro_f1 = sum(per_label[label]["f1"] for label in LABELS) / len(LABELS)

    confusion = {
        gold: {
            pred: sum(1 for g, p in zip(gold_labels, pred_labels) if g == gold and p == pred)
            for pred in LABELS
        }
        for gold in LABELS
    }

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "num_examples": len(gold_labels),
    }


def compute_evidence_metrics(
    gold_evidence_sets: Sequence[set[str]],
    pred_evidence_sets: Sequence[set[str]],
) -> Dict:
    assert len(gold_evidence_sets) == len(pred_evidence_sets), "gold/pred evidence lengths must match"

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []

    for gold_set, pred_set in zip(gold_evidence_sets, pred_evidence_sets):
        tp = len(gold_set & pred_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = f1_score(precision, recall)

        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = f1_score(micro_precision, micro_recall)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0,
        "macro_recall": sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0,
        "macro_f1": sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0,
        "micro_tp": micro_tp,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
        "num_examples": len(gold_evidence_sets),
    }


def evaluate_scifact_predictions(
    queries: Sequence[Query],
    predictions: Sequence[Dict],
) -> Dict:
    pred_by_query_id = {pred["query_id"]: pred for pred in predictions}

    gold_labels: List[str] = []
    pred_labels: List[str] = []

    gold_evidence_sets: List[set[str]] = []
    pred_evidence_sets: List[set[str]] = []

    missing_predictions = 0

    for query in queries:
        pred = pred_by_query_id.get(query.query_id)

        if pred is None:
            missing_predictions += 1
            pred_label = "insufficient"
            pred_evidence = []
        else:
            pred_label = pred.get("predicted_label", "insufficient")
            pred_evidence = pred.get("predicted_evidence_chunks", [])

        gold_label = query.gold_label or "insufficient"
        gold_chunks = {
            ev.chunk_id
            for ev in query.gold_evidence
            if ev.chunk_id is not None
        }

        gold_labels.append(gold_label)
        pred_labels.append(pred_label)

        gold_evidence_sets.append(gold_chunks)
        pred_evidence_sets.append(set(pred_evidence))

    label_metrics = compute_label_metrics(gold_labels, pred_labels)
    evidence_metrics = compute_evidence_metrics(gold_evidence_sets, pred_evidence_sets)

    return {
        "label_metrics": label_metrics,
        "evidence_metrics": evidence_metrics,
        "missing_predictions": missing_predictions,
    }