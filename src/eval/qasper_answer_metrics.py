from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Sequence

from src.graph.schemas import Query


def normalize_answer(text: str | None) -> str:
    if text is None:
        return "unanswerable"

    text = text.strip().lower()

    # map common abstention phrases to one token
    abstain_phrases = {
        "unanswerable",
        "cannot answer",
        "can't answer",
        "not answerable",
        "insufficient information",
        "insufficient evidence",
        "unknown",
        "not enough information",
        "not enough evidence",
        "",
    }
    if text in abstain_phrases:
        return "unanswerable"

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())

    return text if text else "unanswerable"


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_qasper_answer_predictions(
    queries: Sequence[Query],
    predictions: Sequence[Dict],
) -> Dict:
    pred_by_query_id = {pred["query_id"]: pred for pred in predictions}

    em_scores = []
    f1_scores = []
    unanswerable_total = 0
    unanswerable_correct = 0
    missing_predictions = 0

    for query in queries:
        pred = pred_by_query_id.get(query.query_id)
        if pred is None:
            prediction = "unanswerable"
            missing_predictions += 1
        else:
            prediction = pred.get("predicted_answer", "unanswerable")

        gold = query.gold_answer
        if query.metadata.is_unanswerable or not gold:
            gold = "unanswerable"

        em_scores.append(exact_match_score(prediction, gold))
        f1_scores.append(token_f1_score(prediction, gold))

        if normalize_answer(gold) == "unanswerable":
            unanswerable_total += 1
            if normalize_answer(prediction) == "unanswerable":
                unanswerable_correct += 1

    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "unanswerable_accuracy": (
            unanswerable_correct / unanswerable_total if unanswerable_total else 0.0
        ),
        "num_examples": len(queries),
        "missing_predictions": missing_predictions,
    }