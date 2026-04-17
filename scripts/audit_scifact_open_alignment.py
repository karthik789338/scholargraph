from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    claims_path = Path("data/raw/scifact_open/claims.jsonl")
    if not claims_path.exists():
        raise FileNotFoundError(f"Missing file: {claims_path}")

    needed_ids = set()
    num_claims = 0

    for row in read_jsonl(claims_path):
        num_claims += 1
        evidence = row.get("evidence", {})
        if isinstance(evidence, dict):
            for doc_id in evidence.keys():
                needed_ids.add(str(doc_id))

    print(f"Claims loaded: {num_claims}")
    print(f"Unique evidence doc ids in claims: {len(needed_ids)}")

    print("Loading SciFact-Open HF corpus...")
    ds = load_dataset("if-ir/scifact_open", name="corpus", split="corpus")

    sample = ds[0]
    print("HF corpus sample keys:", list(sample.keys()))

    possible_id_keys = ["corpus-id", "doc_id", "id", "corpus_id", "_id", "paper_id"]
    id_key = None
    for k in possible_id_keys:
        if k in sample:
            id_key = k
            break

    if id_key is None:
        raise ValueError(f"Could not find corpus id field in HF sample keys: {list(sample.keys())}")

    corpus_ids = set(str(x) for x in ds[id_key])

    overlap = needed_ids & corpus_ids
    missing = sorted(needed_ids - corpus_ids)

    report = {
        "num_claims": num_claims,
        "num_needed_ids": len(needed_ids),
        "hf_id_key": id_key,
        "num_hf_corpus_ids": len(corpus_ids),
        "num_overlap": len(overlap),
        "overlap_ratio": len(overlap) / len(needed_ids) if needed_ids else 0.0,
        "missing_ids_first_50": missing[:50],
    }

    out_path = Path("data/raw/scifact_open/alignment_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nAlignment report:")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()