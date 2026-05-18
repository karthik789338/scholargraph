from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def agg(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    xs = [float(r[key]) for r in rows]
    return {"mean": mean(xs), "std": pstdev(xs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", default="data/processed/cv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    folds: List[Dict[str, Any]] = []

    for i in range(10):
        fold_dir = cv_root / f"fold{i}"

        flat_summary_path = fold_dir / "summary.json"
        compact_summary_path = fold_dir / "graph_feature_classifier_compact" / "summary.json"

        if not flat_summary_path.exists():
            raise FileNotFoundError(flat_summary_path)
        if not compact_summary_path.exists():
            raise FileNotFoundError(compact_summary_path)

        flat_summary = read_json(str(flat_summary_path))
        compact_summary = read_json(str(compact_summary_path))

        folds.append(
            {
                "fold": f"fold{i}",
                "flat_dev_macro_f1": flat_summary["flat_dev_macro_f1"],
                "flat_dev_evidence_micro_f1": flat_summary["flat_dev_evidence_micro_f1"],
                "compact_dev_macro_f1": compact_summary["dev_label_macro_f1"],
                "compact_dev_evidence_micro_f1": compact_summary["dev_evidence_micro_f1"],
                "delta_macro_f1": compact_summary["dev_label_macro_f1"] - flat_summary["flat_dev_macro_f1"],
                "delta_evidence_micro_f1": compact_summary["dev_evidence_micro_f1"] - flat_summary["flat_dev_evidence_micro_f1"],
            }
        )

    payload = {
        "folds": folds,
        "aggregate": {
            "flat_macro_f1": agg(folds, "flat_dev_macro_f1"),
            "flat_evidence_micro_f1": agg(folds, "flat_dev_evidence_micro_f1"),
            "compact_macro_f1": agg(folds, "compact_dev_macro_f1"),
            "compact_evidence_micro_f1": agg(folds, "compact_dev_evidence_micro_f1"),
            "delta_macro_f1": agg(folds, "delta_macro_f1"),
            "delta_evidence_micro_f1": agg(folds, "delta_evidence_micro_f1"),
        },
    }

    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
