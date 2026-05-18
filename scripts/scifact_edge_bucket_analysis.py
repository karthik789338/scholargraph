from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.eval.scifact_metrics import evaluate_scifact_predictions
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


def write_md(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def index_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["query_id"]): r for r in rows}


def bucket_name(num_edges: int) -> str:
    if num_edges == 0:
        return "0_edges"
    if num_edges == 1:
        return "1_edge"
    return "2plus_edges"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--flat-predictions", required=True)
    parser.add_argument("--compact-predictions", required=True)
    parser.add_argument("--compact-analysis-jsonl", required=True)
    parser.add_argument(
        "--output-json",
        default="reports/scifact_revision/edge_buckets/scifact_edge_bucket_analysis.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/scifact_revision/edge_buckets/scifact_edge_bucket_analysis.md",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    flat_rows = read_jsonl(args.flat_predictions)
    compact_rows = read_jsonl(args.compact_predictions)
    analysis_rows = read_jsonl(args.compact_analysis_jsonl)

    flat_by_qid = index_predictions(flat_rows)
    compact_by_qid = index_predictions(compact_rows)
    edges_by_qid = {str(r["query_id"]): int(r["num_edges"]) for r in analysis_rows}

    buckets: Dict[str, Set[str]] = {
        "0_edges": set(),
        "1_edge": set(),
        "2plus_edges": set(),
    }

    for qid, num_edges in edges_by_qid.items():
        buckets[bucket_name(num_edges)].add(qid)

    summary_rows: List[Dict[str, Any]] = []
    output_payload: Dict[str, Any] = {"buckets": {}}

    for bucket, qids in buckets.items():
        bucket_queries = [q for q in queries if str(q.query_id) in qids]
        bucket_flat = [flat_by_qid[str(q.query_id)] for q in bucket_queries if str(q.query_id) in flat_by_qid]
        bucket_compact = [compact_by_qid[str(q.query_id)] for q in bucket_queries if str(q.query_id) in compact_by_qid]

        flat_metrics = evaluate_scifact_predictions(queries=bucket_queries, predictions=bucket_flat)
        compact_metrics = evaluate_scifact_predictions(queries=bucket_queries, predictions=bucket_compact)

        row = {
            "bucket": bucket,
            "num_queries": len(bucket_queries),
            "flat_macro_f1": flat_metrics["label_metrics"]["macro_f1"],
            "compact_macro_f1": compact_metrics["label_metrics"]["macro_f1"],
            "delta_macro_f1": compact_metrics["label_metrics"]["macro_f1"] - flat_metrics["label_metrics"]["macro_f1"],
            "flat_evidence_micro_f1": flat_metrics["evidence_metrics"]["micro_f1"],
            "compact_evidence_micro_f1": compact_metrics["evidence_metrics"]["micro_f1"],
            "delta_evidence_micro_f1": compact_metrics["evidence_metrics"]["micro_f1"] - flat_metrics["evidence_metrics"]["micro_f1"],
        }
        summary_rows.append(row)

        output_payload["buckets"][bucket] = {
            "num_queries": len(bucket_queries),
            "query_ids": sorted(qids),
            "flat_metrics": flat_metrics,
            "compact_metrics": compact_metrics,
        }

    write_json(args.output_json, output_payload)
    write_md(args.output_md, summary_rows)

    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote MD:   {args.output_md}")


if __name__ == "__main__":
    main()
