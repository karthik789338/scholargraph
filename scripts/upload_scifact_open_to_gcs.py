from __future__ import annotations

import subprocess
from pathlib import Path

BUCKET = "gs://scholar-graph"

UPLOADS = [
    # raw
    ("data/raw/scifact_open/claims.jsonl", f"{BUCKET}/raw/scifact_open/claims/claims.jsonl"),
    ("data/raw/scifact_open/needed_doc_ids.json", f"{BUCKET}/raw/scifact_open/manifests/needed_doc_ids.json"),
    ("data/raw/scifact_open/documents.jsonl", f"{BUCKET}/raw/scifact_open/dev_subset/documents.jsonl"),
    ("data/raw/scifact_open/missing_doc_ids.json", f"{BUCKET}/raw/scifact_open/manifests/missing_doc_ids.json"),

    # processed
    ("data/processed/documents/scifact_open_documents.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/documents/scifact_open_documents.jsonl"),
    ("data/processed/queries/scifact_open.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/queries/scifact_open.jsonl"),
    ("data/processed/queries/scifact_open_with_chunks.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/queries/scifact_open_with_chunks.jsonl"),
    ("data/processed/chunks/scifact_open_chunks.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/chunks/scifact_open_chunks.jsonl"),

    # graph inputs
    ("data/processed/graph_inputs/scifact_open_graph_inputs.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/graph_inputs/scifact_open_graph_inputs.jsonl"),
    ("data/processed/graph_inputs/scifact_open_local_graphs.jsonl", f"{BUCKET}/processed/scifact_open/dev_subset/graph_inputs/scifact_open_local_graphs.jsonl"),

    # results
    ("data/processed/baselines/scifact_open_baseline_predictions.jsonl", f"{BUCKET}/results/scifact_open/dev_subset/scifact_open_baseline_predictions.jsonl"),
    ("data/processed/baselines/scifact_open_baseline_metrics.json", f"{BUCKET}/results/scifact_open/dev_subset/scifact_open_baseline_metrics.json"),
    ("data/processed/graph_outputs/scifact_open_graph_predictions.jsonl", f"{BUCKET}/results/scifact_open/dev_subset/scifact_open_graph_predictions.jsonl"),
    ("data/processed/graph_outputs/scifact_open_graph_metrics.json", f"{BUCKET}/results/scifact_open/dev_subset/scifact_open_graph_metrics.json"),
]

UPLOAD_DIRS = [
    ("data/indexes/bm25/scifact_open", f"{BUCKET}/indexes/scifact_open/dev_subset/bm25/"),
    ("data/indexes/dense/scifact_open", f"{BUCKET}/indexes/scifact_open/dev_subset/dense/"),
]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    # upload files
    for src, dst in UPLOADS:
        src_path = Path(src)
        if not src_path.exists():
            print(f"SKIP missing file: {src}")
            continue
        run(["gcloud", "storage", "cp", str(src_path), dst])

    # upload directories
    for src, dst in UPLOAD_DIRS:
        src_path = Path(src)
        if not src_path.exists():
            print(f"SKIP missing dir: {src}")
            continue
        run(["gcloud", "storage", "cp", "--recursive", str(src_path), dst])

    print("\nDone uploading SciFact-Open dev subset artifacts.")


if __name__ == "__main__":
    main()