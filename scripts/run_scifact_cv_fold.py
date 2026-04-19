from pathlib import Path
import sys
import argparse
import json
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_cmd(cmd):
    logger.info("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SciFact CV fold.")
    parser.add_argument("--fold-name", required=True, help="e.g. fold0")
    parser.add_argument("--train-claims", required=True)
    parser.add_argument("--dev-claims", required=True)
    parser.add_argument("--bm25-index-dir", required=True)
    parser.add_argument("--dense-index-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    args = parser.parse_args()

    py = sys.executable
    fold_name = args.fold_name
    train_name = f"scifact_{fold_name}_train"
    dev_name = f"scifact_{fold_name}_dev"

    cv_dir = ensure_dir(Path("data/processed/cv") / fold_name)
    sweep_dir = ensure_dir(cv_dir / "sweep")

    # 1) Prepare train/dev queries
    run_cmd([
        py, "scripts/prepare_scifact_claims_file.py",
        "--claims-path", args.train_claims,
        "--output-name", train_name,
    ])
    run_cmd([
        py, "scripts/prepare_scifact_claims_file.py",
        "--claims-path", args.dev_claims,
        "--output-name", dev_name,
    ])

    # 2) Attach chunk-level evidence
    run_cmd([
        py, "-m", "src.data.build_queries",
        "--dataset", "scifact",
        "--queries", f"data/processed/queries/{train_name}.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output", f"data/processed/queries/{train_name}_with_chunks.jsonl",
    ])
    run_cmd([
        py, "-m", "src.data.build_queries",
        "--dataset", "scifact",
        "--queries", f"data/processed/queries/{dev_name}.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output", f"data/processed/queries/{dev_name}_with_chunks.jsonl",
    ])

    # 3) Build graph inputs
    run_cmd([
        py, "-m", "src.data.build_graph_inputs",
        "--queries", f"data/processed/queries/{train_name}_with_chunks.jsonl",
        "--output", f"data/processed/graph_inputs/{train_name}_graph_inputs.jsonl",
        "--index-type", "hybrid",
        "--bm25-index-dir", args.bm25_index_dir,
        "--dense-index-dir", args.dense_index_dir,
        "--top-k", str(args.top_k),
        "--restrict-to-candidate-docs",
    ])
    run_cmd([
        py, "-m", "src.data.build_graph_inputs",
        "--queries", f"data/processed/queries/{dev_name}_with_chunks.jsonl",
        "--output", f"data/processed/graph_inputs/{dev_name}_graph_inputs.jsonl",
        "--index-type", "hybrid",
        "--bm25-index-dir", args.bm25_index_dir,
        "--dense-index-dir", args.dense_index_dir,
        "--top-k", str(args.top_k),
        "--restrict-to-candidate-docs",
    ])

    # 4) Build local graphs
    run_cmd([
        py, "-m", "src.graph.build_local_graph",
        "--graph-inputs", f"data/processed/graph_inputs/{train_name}_graph_inputs.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output", f"data/processed/graph_inputs/{train_name}_local_graphs.jsonl",
    ])
    run_cmd([
        py, "-m", "src.graph.build_local_graph",
        "--graph-inputs", f"data/processed/graph_inputs/{dev_name}_graph_inputs.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output", f"data/processed/graph_inputs/{dev_name}_local_graphs.jsonl",
    ])

    # 5) Flat baseline on dev
    flat_metrics_path = cv_dir / "flat_dev_metrics.json"
    run_cmd([
        py, "-m", "src.baselines.scifact_baseline",
        "--queries", f"data/processed/queries/{dev_name}_with_chunks.jsonl",
        "--graph-inputs", f"data/processed/graph_inputs/{dev_name}_graph_inputs.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output-predictions", str(cv_dir / "flat_dev_predictions.jsonl"),
        "--output-metrics", str(flat_metrics_path),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--top-nli-chunks", str(args.top_nli_chunks),
    ])

    # 6) Sweep thresholds on train
    run_cmd([
        py, "scripts/sweep_scifact_graph_thresholds.py",
        "--queries", f"data/processed/queries/{train_name}_with_chunks.jsonl",
        "--graph-inputs", f"data/processed/graph_inputs/{train_name}_graph_inputs.jsonl",
        "--local-graphs", f"data/processed/graph_inputs/{train_name}_local_graphs.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output-dir", str(sweep_dir),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--top-nli-chunks", str(args.top_nli_chunks),
    ])

    best = load_json(sweep_dir / "scifact_dev_graph_best_thresholds.json")

    # 7) Apply best train thresholds to dev
    graph_metrics_path = cv_dir / "graph_dev_metrics.json"
    run_cmd([
        py, "-m", "src.graph.scifact_graph_verdict",
        "--queries", f"data/processed/queries/{dev_name}_with_chunks.jsonl",
        "--graph-inputs", f"data/processed/graph_inputs/{dev_name}_graph_inputs.jsonl",
        "--local-graphs", f"data/processed/graph_inputs/{dev_name}_local_graphs.jsonl",
        "--chunks", "data/processed/chunks/scifact_chunks.jsonl",
        "--output-predictions", str(cv_dir / "graph_dev_predictions.jsonl"),
        "--output-metrics", str(graph_metrics_path),
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
        "--top-nli-chunks", str(args.top_nli_chunks),
        "--label-threshold", str(best["label_threshold"]),
        "--margin-threshold", str(best["margin_threshold"]),
        "--conflict-threshold", str(best["conflict_threshold"]),
        "--neutral-threshold", str(best["neutral_threshold"]),
    ])

    flat = load_json(flat_metrics_path)
    graph = load_json(graph_metrics_path)

    summary = {
        "fold_name": fold_name,
        "best_thresholds": {
            "label_threshold": best["label_threshold"],
            "margin_threshold": best["margin_threshold"],
            "conflict_threshold": best["conflict_threshold"],
            "neutral_threshold": best["neutral_threshold"],
        },
        "flat_dev_macro_f1": flat["label_metrics"]["macro_f1"],
        "flat_dev_evidence_micro_f1": flat["evidence_metrics"]["micro_f1"],
        "graph_dev_macro_f1": graph["label_metrics"]["macro_f1"],
        "graph_dev_evidence_micro_f1": graph["evidence_metrics"]["micro_f1"],
        "delta_macro_f1": graph["label_metrics"]["macro_f1"] - flat["label_metrics"]["macro_f1"],
        "delta_evidence_micro_f1": graph["evidence_metrics"]["micro_f1"] - flat["evidence_metrics"]["micro_f1"],
    }

    write_json(summary, cv_dir / "summary.json")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
