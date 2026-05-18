from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


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


def make_plot(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    taus = [float(r["tau"]) for r in rows]
    macro = [float(r["label_macro_f1"]) for r in rows]
    evid = [float(r["evidence_micro_f1"]) for r in rows]
    avg_edges = [float(r["avg_edges_per_query"]) for r in rows]
    pct_zero = [float(r["pct_queries_zero_edges"]) for r in rows]

    fig = plt.figure(figsize=(10, 4.5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(taus, macro, marker="o", label="Label Macro F1")
    ax1.plot(taus, evid, marker="s", label="Evidence Micro F1")
    ax1.set_xlabel("Tau")
    ax1.set_ylabel("Performance")
    ax1.set_title("SciFact dev performance vs. tau")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(taus, avg_edges, marker="o", label="Avg edges/query")
    ax2.plot(taus, pct_zero, marker="s", label="% zero-edge queries")
    ax2.set_xlabel("Tau")
    ax2.set_ylabel("Graph sparsity")
    ax2.set_title("Graph sparsity vs. tau")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.60, 0.70],
    )
    parser.add_argument(
        "--run-template",
        required=True,
        help=(
            "Shell command template for one compact-model run. "
            "Use {tau}, {tag}, {summary_path}, and {analysis_path} placeholders."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="reports/scifact_revision/tau_sweep/scifact_tau_sweep.json",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/scifact_revision/tau_sweep/scifact_tau_sweep.csv",
    )
    parser.add_argument(
        "--output-md",
        default="reports/scifact_revision/tau_sweep/scifact_tau_sweep.md",
    )
    parser.add_argument(
        "--output-plot",
        default="reports/scifact_revision/tau_sweep/scifact_tau_sweep.png",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []

    for tau in args.taus:
        tag = str(tau).replace(".", "p")
        summary_path = f"reports/scifact_revision/tau_sweep/runs/tau_{tag}/summary.json"
        analysis_path = f"reports/scifact_revision/tau_sweep/runs/tau_{tag}/analysis.jsonl"

        cmd = args.run_template.format(
            tau=tau,
            tag=tag,
            summary_path=summary_path,
            analysis_path=analysis_path,
        )
        print(f"\n=== Running tau={tau:.2f} ===")
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

        summary = load_json(summary_path)
        graph_stats = summary.get("graph_stats", {})

        row = {
            "tau": tau,
            "label_macro_f1": summary["dev_label_macro_f1"],
            "evidence_micro_f1": summary["dev_evidence_micro_f1"],
            "avg_nodes_per_query": graph_stats.get("avg_nodes_per_query", 0.0),
            "avg_edges_per_query": graph_stats.get("avg_edges_per_query", 0.0),
            "pct_queries_zero_edges": graph_stats.get("pct_queries_zero_edges", 0.0),
            "pct_queries_one_edge": graph_stats.get("pct_queries_one_edge", 0.0),
            "pct_queries_two_plus_edges": graph_stats.get("pct_queries_two_plus_edges", 0.0),
            "summary_path": summary_path,
            "analysis_path": analysis_path,
        }
        rows.append(row)

    payload = {"rows": rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    make_plot(args.output_plot, rows)

    print(f"\nWrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")
    print(f"Wrote MD:   {args.output_md}")
    print(f"Wrote PNG:  {args.output_plot}")


if __name__ == "__main__":
    main()
