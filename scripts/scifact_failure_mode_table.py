from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.graph.scifact_graph_verdict import load_queries


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def idx_by_qid(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["query_id"]): r for r in rows}


def get_gold_evidence_chunks(query: Any) -> List[str]:
    if hasattr(query, "gold_evidence_chunks"):
        return list(getattr(query, "gold_evidence_chunks") or [])
    if isinstance(query, dict):
        return list(query.get("gold_evidence_chunks", []) or [])
    return []


def get_gold_label(query: Any) -> str:
    if hasattr(query, "gold_label"):
        return str(getattr(query, "gold_label"))
    if isinstance(query, dict):
        return str(query.get("gold_label"))
    raise ValueError("Could not read gold label from query object.")


def get_query_id(query: Any) -> str:
    if hasattr(query, "query_id"):
        return str(getattr(query, "query_id"))
    if isinstance(query, dict):
        return str(query.get("query_id"))
    raise ValueError("Could not read query id from query object.")


def get_query_text(query: Any) -> str:
    if hasattr(query, "text"):
        return str(getattr(query, "text"))
    if isinstance(query, dict):
        return str(query.get("text", ""))
    return ""


def edge_bucket(num_edges: int) -> str:
    if num_edges <= 0:
        return "0_edges"
    if num_edges == 1:
        return "1_edge"
    return "2plus_edges"


def claim_tags(text: str) -> List[str]:
    tags: List[str] = []
    t = text.lower()

    if re.search(r"\d", text):
        tags.append("numerical")
    if any(tok in t for tok in ["not ", " no ", "without ", "never ", "none ", "neither "]):
        tags.append("negation")
    if any(tok in t for tok in ["increase", "decrease", "improve", "impair", "reduce", "promote", "inhibit"]):
        tags.append("directional")
    if any(tok in t for tok in [" than ", " compared ", " versus ", " vs "]):
        tags.append("comparative")
    if not tags:
        tags.append("other")
    return tags


def classify_failure(
    gold_label: str,
    pred_label: str,
    gold_evidence: Sequence[str],
    pred_evidence: Sequence[str],
) -> str:
    gold_set = set(gold_evidence)
    pred_set = set(pred_evidence)
    overlap = gold_set & pred_set

    if pred_label != gold_label:
        if gold_label in {"supports", "refutes"} and pred_label == "insufficient":
            return "verifiable_to_insufficient"
        if gold_label == "insufficient" and pred_label in {"supports", "refutes"}:
            return "insufficient_to_asserted"
        if gold_label in {"supports", "refutes"} and pred_label in {"supports", "refutes"}:
            return "support_refute_confusion"
        return "other_verdict_error"

    # verdict correct
    if gold_label == "insufficient":
        return "correct_insufficient"

    if len(pred_set) == 0 or len(overlap) == 0:
        return "correct_label_evidence_miss"

    if pred_set == gold_set:
        return "correct_label_exact_evidence"

    return "correct_label_partial_evidence"


def transition_label(
    gold_label: str,
    flat_label: str,
    compact_label: str,
) -> str:
    flat_ok = (flat_label == gold_label)
    compact_ok = (compact_label == gold_label)

    if flat_ok and compact_ok:
        return "both_correct"
    if (not flat_ok) and compact_ok:
        return "compact_fixes_flat"
    if flat_ok and (not compact_ok):
        return "compact_hurts_flat"
    return "both_wrong"


def top_examples(rows: Sequence[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["failure_mode"]].append(row)

    for mode, items in grouped.items():
        for row in items[:n]:
            out.append(row)
    return out


def write_md(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("# SciFact failure mode frequency table")
    lines.append("")

    # Transition table
    lines.append("## Flat vs compact transition summary")
    lines.append("")
    lines.append("| transition | count | pct_of_dev |")
    lines.append("| --- | ---: | ---: |")
    for row in payload["transition_rows"]:
        lines.append(f"| {row['transition']} | {row['count']} | {row['pct_of_dev']:.4f} |")

    lines.append("")
    lines.append("## Compact failure / success categories")
    lines.append("")
    lines.append("| failure_mode | count | pct_of_dev | pct_of_compact_errors |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in payload["failure_rows"]:
        lines.append(
            f"| {row['failure_mode']} | {row['count']} | {row['pct_of_dev']:.4f} | {row['pct_of_compact_errors']:.4f} |"
        )

    lines.append("")
    lines.append("## Failure modes by edge bucket")
    lines.append("")
    lines.append("| edge_bucket | failure_mode | count |")
    lines.append("| --- | --- | ---: |")
    for row in payload["failure_by_edge_rows"]:
        lines.append(f"| {row['edge_bucket']} | {row['failure_mode']} | {row['count']} |")

    lines.append("")
    lines.append("## Failure modes by claim tag")
    lines.append("")
    lines.append("| claim_tag | failure_mode | count |")
    lines.append("| --- | --- | ---: |")
    for row in payload["failure_by_tag_rows"]:
        lines.append(f"| {row['claim_tag']} | {row['failure_mode']} | {row['count']} |")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SciFact failure mode frequency tables.")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--flat-predictions", required=True)
    parser.add_argument("--compact-predictions", required=True)
    parser.add_argument("--compact-analysis-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--examples-jsonl", required=True)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    flat_rows = read_jsonl(args.flat_predictions)
    compact_rows = read_jsonl(args.compact_predictions)
    analysis_rows = read_jsonl(args.compact_analysis_jsonl)

    flat_by_qid = idx_by_qid(flat_rows)
    compact_by_qid = idx_by_qid(compact_rows)
    analysis_by_qid = idx_by_qid(analysis_rows)

    all_rows: List[Dict[str, Any]] = []
    transition_counter = Counter()
    failure_counter = Counter()
    failure_by_edge = Counter()
    failure_by_tag = Counter()

    compact_error_total = 0

    for query in queries:
        qid = get_query_id(query)
        qtext = get_query_text(query)
        gold_label = get_gold_label(query)
        gold_evidence = get_gold_evidence_chunks(query)

        flat_pred = flat_by_qid[qid]
        compact_pred = compact_by_qid[qid]
        analysis = analysis_by_qid[qid]

        flat_label = str(flat_pred["predicted_label"])
        compact_label = str(compact_pred["predicted_label"])

        flat_evidence = list(flat_pred.get("predicted_evidence_chunks", []) or [])
        compact_evidence = list(compact_pred.get("predicted_evidence_chunks", []) or [])

        mode = classify_failure(
            gold_label=gold_label,
            pred_label=compact_label,
            gold_evidence=gold_evidence,
            pred_evidence=compact_evidence,
        )
        transition = transition_label(
            gold_label=gold_label,
            flat_label=flat_label,
            compact_label=compact_label,
        )

        num_edges = int(analysis.get("num_edges", 0))
        bucket = edge_bucket(num_edges)
        tags = claim_tags(qtext)

        row = {
            "query_id": qid,
            "query_text": qtext,
            "gold_label": gold_label,
            "flat_label": flat_label,
            "compact_label": compact_label,
            "gold_evidence_chunks": gold_evidence,
            "flat_predicted_evidence_chunks": flat_evidence,
            "compact_predicted_evidence_chunks": compact_evidence,
            "num_edges": num_edges,
            "edge_bucket": bucket,
            "claim_tags": tags,
            "failure_mode": mode,
            "transition": transition,
        }
        all_rows.append(row)

        transition_counter[transition] += 1
        failure_counter[mode] += 1

        if mode not in {"correct_insufficient", "correct_label_exact_evidence"}:
            compact_error_total += 1

        for tag in tags:
            failure_by_tag[(tag, mode)] += 1
        failure_by_edge[(bucket, mode)] += 1

    n_dev = len(all_rows)

    transition_rows = []
    for key, count in sorted(transition_counter.items()):
        transition_rows.append({
            "transition": key,
            "count": count,
            "pct_of_dev": pct(count, n_dev),
        })

    failure_rows = []
    for key, count in sorted(failure_counter.items(), key=lambda x: (-x[1], x[0])):
        failure_rows.append({
            "failure_mode": key,
            "count": count,
            "pct_of_dev": pct(count, n_dev),
            "pct_of_compact_errors": pct(count, compact_error_total),
        })

    failure_by_edge_rows = []
    for (bucket, mode), count in sorted(failure_by_edge.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        failure_by_edge_rows.append({
            "edge_bucket": bucket,
            "failure_mode": mode,
            "count": count,
        })

    failure_by_tag_rows = []
    for (tag, mode), count in sorted(failure_by_tag.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        failure_by_tag_rows.append({
            "claim_tag": tag,
            "failure_mode": mode,
            "count": count,
        })

    examples = top_examples(all_rows, n=5)

    payload = {
        "n_dev": n_dev,
        "compact_error_total": compact_error_total,
        "transition_rows": transition_rows,
        "failure_rows": failure_rows,
        "failure_by_edge_rows": failure_by_edge_rows,
        "failure_by_tag_rows": failure_by_tag_rows,
    }

    write_json(args.output_json, payload)
    write_md(args.output_md, payload)
    write_jsonl(args.examples_jsonl, examples)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
