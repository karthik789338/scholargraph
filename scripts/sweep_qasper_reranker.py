from pathlib import Path
import json
import itertools
import subprocess
import sys

def run(cmd):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    py = sys.executable

    queries = "data/processed/queries/qasper_validation_with_chunks.jsonl"
    graph_inputs = "data/processed/graph_inputs/qasper_validation_graph_inputs.jsonl"
    local_graphs = "data/processed/graph_inputs/qasper_validation_local_graphs.jsonl"
    dense_index_dir = "data/indexes/dense/qasper_validation"
    chunks = "data/processed/chunks/qasper_validation_chunks.jsonl"

    out_dir = Path("data/processed/qasper_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_weights = [0.4, 0.5, 0.6]
    retrieval_weights = [0.2, 0.3, 0.4]
    graph_weights = [0.1, 0.2, 0.3]
    mmr_lambdas = [0.6, 0.7, 0.8]

    results = []
    run_idx = 0
    total = (
        len(semantic_weights)
        * len(retrieval_weights)
        * len(graph_weights)
        * len(mmr_lambdas)
    )

    for semantic_weight, retrieval_weight, graph_weight, mmr_lambda in itertools.product(
        semantic_weights,
        retrieval_weights,
        graph_weights,
        mmr_lambdas,
    ):
        run_idx += 1
        tag = (
            f"sw{semantic_weight:.1f}_"
            f"rw{retrieval_weight:.1f}_"
            f"gw{graph_weight:.1f}_"
            f"mmr{mmr_lambda:.1f}"
        )

        evidence_pred = out_dir / f"{tag}_evidence_predictions.jsonl"
        evidence_metrics = out_dir / f"{tag}_evidence_metrics.json"
        answer_pred = out_dir / f"{tag}_answer_predictions.jsonl"
        answer_metrics = out_dir / f"{tag}_answer_metrics.json"

        print(f"\n=== Sweep {run_idx}/{total}: {tag} ===")

        run([
            py, "-m", "src.graph.qasper_graph_reranker",
            "--queries", queries,
            "--graph-inputs", graph_inputs,
            "--local-graphs", local_graphs,
            "--dense-index-dir", dense_index_dir,
            "--output-predictions", str(evidence_pred),
            "--output-metrics", str(evidence_metrics),
            "--top-k", "5",
            "--semantic-weight", str(semantic_weight),
            "--retrieval-weight", str(retrieval_weight),
            "--graph-weight", str(graph_weight),
            "--mmr-lambda", str(mmr_lambda),
        ])

        run([
            py, "-m", "src.baselines.qasper_answer_baseline",
            "--queries", queries,
            "--chunks", chunks,
            "--evidence-predictions", str(evidence_pred),
            "--output-predictions", str(answer_pred),
            "--output-metrics", str(answer_metrics),
            "--batch-size", "16",
        ])

        e = load_json(str(evidence_metrics))
        a = load_json(str(answer_metrics))

        row = {
            "tag": tag,
            "semantic_weight": semantic_weight,
            "retrieval_weight": retrieval_weight,
            "graph_weight": graph_weight,
            "mmr_lambda": mmr_lambda,
            "evidence_micro_f1": e.get("micro_f1", 0.0),
            "evidence_micro_precision": e.get("micro_precision", 0.0),
            "evidence_micro_recall": e.get("micro_recall", 0.0),
            "answer_exact_match": a.get("exact_match", 0.0),
            "answer_token_f1": a.get("token_f1", 0.0),
            "unanswerable_accuracy": a.get("unanswerable_accuracy", 0.0),
        }
        results.append(row)

    results_sorted = sorted(
        results,
        key=lambda x: (
            x["answer_token_f1"],
            x["answer_exact_match"],
            x["evidence_micro_f1"],
            x["unanswerable_accuracy"],
        ),
        reverse=True,
    )

    with (out_dir / "qasper_reranker_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2)

    best = results_sorted[0]
    with (out_dir / "qasper_reranker_best.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nTop 10 settings:")
    for row in results_sorted[:10]:
        print(json.dumps(row, indent=2))

    print("\nBest setting:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
