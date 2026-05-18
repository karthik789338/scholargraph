from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

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


def run_cmd(cmd: List[str]) -> None:
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep graph-construction thresholds for SciFact compact classifier.")
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--train-graph-inputs", required=True)
    parser.add_argument("--dev-queries", required=True)
    parser.add_argument("--dev-graph-inputs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-nli-chunks", type=int, default=3)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)

    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.45, 0.55],
        help="Graph edge thresholds to sweep.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for tau in args.taus:
        tag = str(tau).replace(".", "p")

        tau_dir = output_dir / f"tau_{tag}"
        tau_dir.mkdir(parents=True, exist_ok=True)

        train_local_graphs = tau_dir / "train_local_graphs.jsonl"
        dev_local_graphs = tau_dir / "dev_local_graphs.jsonl"
        analysis_jsonl = tau_dir / "dev_analysis.jsonl"
        summary_json = tau_dir / "summary.json"

        run_cmd([
            "python", "-m", "src.graph.build_local_graph",
            "--graph-inputs", args.train_graph_inputs,
            "--chunks", args.chunks,
            "--output", str(train_local_graphs),
            "--edge-threshold", str(tau),
        ])

        run_cmd([
            "python", "-m", "src.graph.build_local_graph",
            "--graph-inputs", args.dev_graph_inputs,
            "--chunks", args.chunks,
            "--output", str(dev_local_graphs),
            "--edge-threshold", str(tau),
        ])

        run_cmd([
            "python", "-m", "src.graph.scifact_graph_feature_classifier",
            "--train-queries", args.train_queries,
            "--train-graph-inputs", args.train_graph_inputs,
            "--train-local-graphs", str(train_local_graphs),
            "--dev-queries", args.dev_queries,
            "--dev-graph-inputs", args.dev_graph_inputs,
            "--dev-local-graphs", str(dev_local_graphs),
            "--chunks", args.chunks,
            "--output-dir", str(tau_dir),
            "--model-name", args.model_name,
            "--batch-size", str(args.batch_size),
            "--max-length", str(args.max_length),
            "--top-nli-chunks", str(args.top_nli_chunks),
            "--max-evidence-chunks", str(args.max_evidence_chunks),
            "--edge-threshold", str(tau),
            "--analysis-jsonl-out", str(analysis_jsonl),
        ])

        summary = load_json(str(summary_json))
        graph_stats = summary.get("graph_stats", {})

        rows.append({
            "tau": tau,
            "label_macro_f1": summary["dev_label_macro_f1"],
            "evidence_micro_f1": summary["dev_evidence_micro_f1"],
            "avg_nodes_per_query": graph_stats.get("avg_nodes_per_query", 0.0),
            "avg_edges_per_query": graph_stats.get("avg_edges_per_query", 0.0),
            "pct_queries_zero_edges": graph_stats.get("pct_queries_zero_edges", 0.0),
            "pct_queries_one_edge": graph_stats.get("pct_queries_one_edge", 0.0),
            "pct_queries_two_plus_edges": graph_stats.get("pct_queries_two_plus_edges", 0.0),
            "summary_json": str(summary_json),
            "analysis_jsonl": str(analysis_jsonl),
        })

    payload = {"rows": rows}
    write_json(str(output_dir / "scifact_tau_graph_sweep.json"), payload)
    write_csv(str(output_dir / "scifact_tau_graph_sweep.csv"), rows)
    write_md(str(output_dir / "scifact_tau_graph_sweep.md"), rows)

    print(f"\nWrote {output_dir / 'scifact_tau_graph_sweep.json'}")
    print(f"Wrote {output_dir / 'scifact_tau_graph_sweep.csv'}")
    print(f"Wrote {output_dir / 'scifact_tau_graph_sweep.md'}")


if __name__ == "__main__":
    main()
