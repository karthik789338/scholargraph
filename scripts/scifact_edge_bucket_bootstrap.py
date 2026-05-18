from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.scifact_graph_verdict import load_queries


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
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


def idx_by_qid(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["query_id"]): r for r in rows}


def edge_bucket(num_edges: int) -> str:
    if num_edges <= 0:
        return "0_edges"
    if num_edges == 1:
        return "1_edge"
    return "2plus_edges"


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--flat-predictions", required=True)
    parser.add_argument("--compact-predictions", required=True)
    parser.add_argument("--compact-analysis-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    flat_rows = read_jsonl(args.flat_predictions)
    compact_rows = read_jsonl(args.compact_predictions)
    analysis_rows = read_jsonl(args.compact_analysis_jsonl)

    flat_by_qid = idx_by_qid(flat_rows)
    compact_by_qid = idx_by_qid(compact_rows)
    edge_by_qid = {str(r["query_id"]): int(r.get("num_edges", 0)) for r in analysis_rows}

    buckets = {
        "0_edges": [],
        "1_edge": [],
        "2plus_edges": [],
    }

    for q in queries:
        qid = str(q.query_id)
        buckets[edge_bucket(edge_by_qid.get(qid, 0))].append(q)

    rng = random.Random(args.seed)
    output: Dict[str, Any] = {}

    for bucket, bucket_queries in buckets.items():
        n = len(bucket_queries)
        macro_deltas: List[float] = []
        evidence_deltas: List[float] = []

        for _ in range(args.bootstrap_samples):
            sampled = [bucket_queries[rng.randrange(n)] for _ in range(n)] if n > 0 else []

            flat_preds = [flat_by_qid[str(q.query_id)] for q in sampled]
            compact_preds = [compact_by_qid[str(q.query_id)] for q in sampled]

            flat_metrics = evaluate_scifact_predictions(sampled, flat_preds)
            compact_metrics = evaluate_scifact_predictions(sampled, compact_preds)

            macro_delta = compact_metrics["label_metrics"]["macro_f1"] - flat_metrics["label_metrics"]["macro_f1"]
            evidence_delta = compact_metrics["evidence_metrics"]["micro_f1"] - flat_metrics["evidence_metrics"]["micro_f1"]

            macro_deltas.append(float(macro_delta))
            evidence_deltas.append(float(evidence_delta))

        output[bucket] = {
            "num_queries": n,
            "macro_f1_delta": {
                "mean": float(np.mean(macro_deltas)) if macro_deltas else 0.0,
                "ci95_low": percentile(macro_deltas, 2.5) if macro_deltas else 0.0,
                "ci95_high": percentile(macro_deltas, 97.5) if macro_deltas else 0.0,
            },
            "evidence_micro_f1_delta": {
                "mean": float(np.mean(evidence_deltas)) if evidence_deltas else 0.0,
                "ci95_low": percentile(evidence_deltas, 2.5) if evidence_deltas else 0.0,
                "ci95_high": percentile(evidence_deltas, 97.5) if evidence_deltas else 0.0,
            },
        }

    write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
