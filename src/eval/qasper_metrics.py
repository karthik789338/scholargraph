from __future__ import annotations

from typing import Dict, Sequence


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_evidence_metrics(
    gold_evidence_sets: Sequence[set[str]],
    pred_evidence_sets: Sequence[set[str]],
) -> Dict:
    assert len(gold_evidence_sets) == len(pred_evidence_sets)

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

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