from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
from typing import List


def pick_one(fold_dir: Path, patterns: List[str], label: str) -> str:
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(str(fold_dir / pat)))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"[{fold_dir.name}] could not find {label} with patterns: {patterns}")
    if len(matches) > 1:
        print(f"[{fold_dir.name}] multiple matches for {label}, picking first:")
        for m in matches:
            print("   ", m)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact SciFact CV using existing fold artifacts.")
    parser.add_argument("--cv-root", default="data/processed/cv")
    parser.add_argument("--output-root", default="reports/scifact_cv10_compact")
    parser.add_argument("--chunks", default="data/processed/chunks/scifact_chunks.jsonl")
    parser.add_argument("--folds", nargs="+", default=[str(i) for i in range(10)])
    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)
    parser.add_argument("--edge-threshold", type=float, default=0.50)
    args = parser.parse_args()

    cv_root = Path(args.cv_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for fold_id in args.folds:
        fold_name = f"fold{fold_id}"
        fold_dir = cv_root / fold_name
        if not fold_dir.exists():
            raise FileNotFoundError(fold_dir)

        train_queries = pick_one(
            fold_dir,
            ["*train*with_chunks*.jsonl", "*train*queries*.jsonl", "*queries_train*.jsonl"],
            "train queries",
        )
        dev_queries = pick_one(
            fold_dir,
            ["*dev*with_chunks*.jsonl", "*dev*queries*.jsonl", "*queries_dev*.jsonl"],
            "dev queries",
        )
        train_graph_inputs = pick_one(
            fold_dir,
            ["*train*graph_inputs*.jsonl", "*graph_inputs_train*.jsonl"],
            "train graph inputs",
        )
        dev_graph_inputs = pick_one(
            fold_dir,
            ["*dev*graph_inputs*.jsonl", "*graph_inputs_dev*.jsonl"],
            "dev graph inputs",
        )
        train_local_graphs = pick_one(
            fold_dir,
            ["*train*local_graphs*.jsonl", "*local_graphs_train*.jsonl"],
            "train local graphs",
        )
        dev_local_graphs = pick_one(
            fold_dir,
            ["*dev*local_graphs*.jsonl", "*local_graphs_dev*.jsonl"],
            "dev local graphs",
        )

        fold_out = output_root / fold_name
        fold_out.mkdir(parents=True, exist_ok=True)

        analysis_jsonl = fold_out / "dev_analysis.jsonl"

        print(f"\n=== {fold_name} ===")
        print("train_queries     =", train_queries)
        print("dev_queries       =", dev_queries)
        print("train_graph_inputs=", train_graph_inputs)
        print("dev_graph_inputs  =", dev_graph_inputs)
        print("train_local_graphs=", train_local_graphs)
        print("dev_local_graphs  =", dev_local_graphs)

        cmd = [
            "python", "-m", "src.graph.scifact_graph_feature_classifier",
            "--train-queries", train_queries,
            "--train-graph-inputs", train_graph_inputs,
            "--train-local-graphs", train_local_graphs,
            "--dev-queries", dev_queries,
            "--dev-graph-inputs", dev_graph_inputs,
            "--dev-local-graphs", dev_local_graphs,
            "--chunks", args.chunks,
            "--output-dir", str(fold_out),
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--max-length", str(args.max_length),
            "--top-nli-chunks", str(args.top_nli_chunks),
            "--max-evidence-chunks", str(args.max_evidence_chunks),
            "--edge-threshold", str(args.edge_threshold),
            "--analysis-jsonl-out", str(analysis_jsonl),
        ]
        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        summary_path = fold_out / "summary.json"
        print(f"Wrote {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
