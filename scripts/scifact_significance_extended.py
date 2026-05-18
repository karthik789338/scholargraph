from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.scifact_graph_verdict import load_queries

try:
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def prediction_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["query_id"]): r for r in rows}


def subset_predictions(
    queries,
    pred_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [pred_index[str(q.query_id)] for q in queries if str(q.query_id) in pred_index]


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    arr = np.asarray(xs, dtype=float)
    return float(np.percentile(arr, q))


def paired_t_and_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": len(a),
        "a_mean": float(np.mean(a)) if a else 0.0,
        "b_mean": float(np.mean(b)) if b else 0.0,
        "delta_mean": float(np.mean(np.asarray(b) - np.asarray(a))) if a else 0.0,
    }

    if not SCIPY_OK:
        result["scipy_available"] = False
        return result

    t_stat, t_p = ttest_rel(b, a)
    result["scipy_available"] = True
    result["paired_t_test"] = {
        "t_stat": float(t_stat),
        "p_value": float(t_p),
    }

    diffs = np.asarray(b) - np.asarray(a)
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        result["wilcoxon"] = {
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "all paired differences are zero",
        }
    else:
        w_stat, w_p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
        result["wilcoxon"] = {
            "statistic": float(w_stat),
            "p_value": float(w_p),
        }

    return result


def bootstrap_dev_delta(
    queries,
    flat_index: Dict[str, Dict[str, Any]],
    compact_index: Dict[str, Dict[str, Any]],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    query_list = list(queries)
    n = len(query_list)

    macro_deltas: List[float] = []
    evidence_deltas: List[float] = []

    for _ in range(n_boot):
        sampled = [query_list[rng.randrange(n)] for _ in range(n)]

        flat_preds = subset_predictions(sampled, flat_index)
        compact_preds = subset_predictions(sampled, compact_index)

        flat_metrics = evaluate_scifact_predictions(sampled, flat_preds)
        compact_metrics = evaluate_scifact_predictions(sampled, compact_preds)

        macro_delta = (
            compact_metrics["label_metrics"]["macro_f1"]
            - flat_metrics["label_metrics"]["macro_f1"]
        )
        evidence_delta = (
            compact_metrics["evidence_metrics"]["micro_f1"]
            - flat_metrics["evidence_metrics"]["micro_f1"]
        )

        macro_deltas.append(float(macro_delta))
        evidence_deltas.append(float(evidence_delta))

    return {
        "n_bootstrap": n_boot,
        "macro_f1_delta": {
            "mean": float(np.mean(macro_deltas)),
            "ci95_low": percentile(macro_deltas, 2.5),
            "ci95_high": percentile(macro_deltas, 97.5),
        },
        "evidence_micro_f1_delta": {
            "mean": float(np.mean(evidence_deltas)),
            "ci95_low": percentile(evidence_deltas, 2.5),
            "ci95_high": percentile(evidence_deltas, 97.5),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended significance testing for SciFact flat vs compact.")
    parser.add_argument("--cv-json", required=True, help="Path to 5-fold CV aggregate JSON")
    parser.add_argument("--queries", required=True, help="Original dev queries JSONL")
    parser.add_argument("--flat-predictions", required=True, help="Flat dev predictions JSONL")
    parser.add_argument("--compact-predictions", required=True, help="Compact dev predictions JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cv = read_json(args.cv_json)
    folds = cv["folds"]

    flat_macro = [float(row["flat_dev_macro_f1"]) for row in folds]
    compact_macro = [float(row["compact_dev_macro_f1"]) for row in folds]

    flat_evidence = [float(row["flat_dev_evidence_micro_f1"]) for row in folds]
    compact_evidence = [float(row["compact_dev_evidence_micro_f1"]) for row in folds]

    fold_tests = {
        "macro_f1": paired_t_and_wilcoxon(flat_macro, compact_macro),
        "evidence_micro_f1": paired_t_and_wilcoxon(flat_evidence, compact_evidence),
    }

    queries = load_queries(args.queries)
    flat_rows = read_jsonl(args.flat_predictions)
    compact_rows = read_jsonl(args.compact_predictions)

    flat_index = prediction_index(flat_rows)
    compact_index = prediction_index(compact_rows)

    bootstrap = bootstrap_dev_delta(
        queries=queries,
        flat_index=flat_index,
        compact_index=compact_index,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )

    out = {
        "cv_fold_tests": fold_tests,
        "dev_bootstrap": bootstrap,
        "inputs": {
            "cv_json": args.cv_json,
            "queries": args.queries,
            "flat_predictions": args.flat_predictions,
            "compact_predictions": args.compact_predictions,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
    }

    write_json(args.output, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
