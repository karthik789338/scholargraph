from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def normalize_gold_label(label: str) -> str:
    label = str(label).strip().upper()
    if label in {"SUPPORT", "SUPPORTS"}:
        return "SUPPORT"
    if label in {"CONTRADICT", "REFUTE", "REFUTES"}:
        return "CONTRADICT"
    return label


def normalize_pred_label(label: str) -> str:
    return normalize_gold_label(label)


def normalize_doc_id(doc_id: Any) -> str:
    doc_id = str(doc_id).strip()
    if doc_id.startswith("scifact_doc_"):
        doc_id = doc_id[len("scifact_doc_"):]
    return doc_id


def build_gold_index(gold_claims: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for claim in gold_claims:
        cid = int(claim["id"])
        gold_evidence = claim.get("evidence", {}) or {}

        doc_map: Dict[str, Dict[str, Any]] = {}
        for doc_id, evidence_sets in gold_evidence.items():
            evidence_sets = evidence_sets or []
            if not evidence_sets:
                continue

            label = normalize_gold_label(evidence_sets[0]["label"])
            sentence_sets = [set(ev["sentences"]) for ev in evidence_sets]
            all_gold_sentences = set()
            for s in sentence_sets:
                all_gold_sentences |= s

            doc_map[normalize_doc_id(doc_id)] = {
                "label": label,
                "sentence_sets": sentence_sets,
                "all_gold_sentences": all_gold_sentences,
            }

        out[cid] = {
            "docs": doc_map
        }
    return out


def build_pred_index(pred_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in pred_rows:
        cid = int(row["id"])
        pred_evidence = row.get("evidence", {}) or {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for doc_id, info in pred_evidence.items():
            label = normalize_pred_label(info.get("label", ""))
            sentences = [int(x) for x in info.get("sentences", [])]
            doc_map[normalize_doc_id(doc_id)] = {
                "label": label,
                "sentences": sentences,
                "sentence_set": set(sentences),
                "first3_set": set(sentences[:3]),
            }

        out[cid] = {
            "docs": doc_map
        }
    return out


def evaluate(gold_idx: Dict[int, Dict[str, Any]], pred_idx: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    # Abstract Label+Rationale
    gold_abs = 0
    pred_abs = 0
    correct_abs_label_only = 0
    correct_abs_label_rationale = 0

    # Sentence Selection+Label
    gold_sent = 0
    pred_sent = 0
    correct_sent_label = 0

    for claim_id, gold_claim in gold_idx.items():
        gold_docs = gold_claim["docs"]
        pred_docs = pred_idx.get(claim_id, {"docs": {}})["docs"]

        gold_abs += len(gold_docs)
        pred_abs += len(pred_docs)

        for doc_id, gold_doc in gold_docs.items():
            gold_sent += len(gold_doc["all_gold_sentences"])

        for doc_id, pred_doc in pred_docs.items():
            pred_sent += len(pred_doc["sentence_set"])

            if doc_id not in gold_docs:
                continue

            gold_doc = gold_docs[doc_id]

            # label-only abstract correctness
            if pred_doc["label"] == gold_doc["label"]:
                correct_abs_label_only += 1

                # label+rationale abstract correctness
                pred_first3 = pred_doc["first3_set"]
                if any(sentence_set.issubset(pred_first3) for sentence_set in gold_doc["sentence_sets"]):
                    correct_abs_label_rationale += 1

                # sentence-level label+selection correctness
                for s in pred_doc["sentence_set"]:
                    for gold_set in gold_doc["sentence_sets"]:
                        if s in gold_set and gold_set.issubset(pred_doc["sentence_set"]):
                            correct_sent_label += 1
                            break

    abstract_label_only_precision = safe_div(correct_abs_label_only, pred_abs)
    abstract_label_only_recall = safe_div(correct_abs_label_only, gold_abs)
    abstract_label_only_f1 = f1(abstract_label_only_precision, abstract_label_only_recall)

    abstract_label_rationale_precision = safe_div(correct_abs_label_rationale, pred_abs)
    abstract_label_rationale_recall = safe_div(correct_abs_label_rationale, gold_abs)
    abstract_label_rationale_f1 = f1(abstract_label_rationale_precision, abstract_label_rationale_recall)

    sentence_selection_label_precision = safe_div(correct_sent_label, pred_sent)
    sentence_selection_label_recall = safe_div(correct_sent_label, gold_sent)
    sentence_selection_label_f1 = f1(sentence_selection_label_precision, sentence_selection_label_recall)

    return {
        "abstract_label_only": {
            "precision": abstract_label_only_precision,
            "recall": abstract_label_only_recall,
            "f1": abstract_label_only_f1,
        },
        "abstract_label_rationale": {
            "precision": abstract_label_rationale_precision,
            "recall": abstract_label_rationale_recall,
            "f1": abstract_label_rationale_f1,
        },
        "sentence_selection_label": {
            "precision": sentence_selection_label_precision,
            "recall": sentence_selection_label_recall,
            "f1": sentence_selection_label_f1,
        },
        "counts": {
            "gold_abstracts": gold_abs,
            "predicted_abstracts": pred_abs,
            "correct_abstracts_label_only": correct_abs_label_only,
            "correct_abstracts_label_rationale": correct_abs_label_rationale,
            "gold_sentences": gold_sent,
            "predicted_sentences": pred_sent,
            "correct_sentences_selection_label": correct_sent_label,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-claims", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold_claims = read_jsonl(args.gold_claims)
    pred_rows = read_jsonl(args.predictions)

    gold_idx = build_gold_index(gold_claims)
    pred_idx = build_pred_index(pred_rows)
    metrics = evaluate(gold_idx, pred_idx)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
