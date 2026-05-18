from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_scifact_label(row: dict) -> str:
    evidence = row.get("evidence", {}) or {}

    if not evidence:
        return "insufficient"

    observed = set()
    for _, ev_sets in evidence.items():
        if not ev_sets:
            continue
        for ev in ev_sets:
            raw = str(ev.get("label", "")).strip().lower()
            if raw in {"support", "supports"}:
                observed.add("supports")
            elif raw in {"contradict", "contradiction", "refute", "refutes"}:
                observed.add("refutes")

    if not observed:
        return "insufficient"
    if len(observed) > 1:
        raise ValueError(f"Claim {row.get('id')} has mixed evidence labels: {observed}")
    return next(iter(observed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified 10-fold SciFact claim splits.")
    parser.add_argument("--claims", required=True, help="Path to SciFact training claims JSONL")
    parser.add_argument("--output-root", required=True, help="Output directory root")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    claims = read_jsonl(args.claims)

    by_label: Dict[str, List[dict]] = defaultdict(list)
    label_counts = Counter()

    for row in claims:
        label = infer_scifact_label(row)
        row["_derived_label"] = label
        by_label[label].append(row)
        label_counts[label] += 1

    rng = random.Random(args.seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    folds: List[List[dict]] = [[] for _ in range(args.n_folds)]
    for label, label_rows in by_label.items():
        for idx, row in enumerate(label_rows):
            folds[idx % args.n_folds].append(row)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"label_counts": dict(label_counts)}, indent=2))

    for i in range(args.n_folds):
        dev_rows = []
        train_rows = []

        for j in range(args.n_folds):
            bucket = folds[j]
            cleaned = []
            for row in bucket:
                row_copy = dict(row)
                row_copy.pop("_derived_label", None)
                cleaned.append(row_copy)
            if j == i:
                dev_rows.extend(cleaned)
            else:
                train_rows.extend(cleaned)

        fold_dir = output_root / f"fold_{i}"
        write_jsonl(fold_dir / "claims_train.jsonl", train_rows)
        write_jsonl(fold_dir / "claims_dev.jsonl", dev_rows)

        print(
            json.dumps(
                {
                    "fold": i,
                    "train": len(train_rows),
                    "dev": len(dev_rows),
                }
            )
        )


if __name__ == "__main__":
    main()
