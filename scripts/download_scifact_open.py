from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def main():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Please install datasets first: pip install datasets"
        ) from e

    target_dir = Path("data/raw/scifact_open")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SciFact-Open claims split from Hugging Face...")
    ds = load_dataset("davidheineman/scifact-open", split="claims")

    claims_out = target_dir / "claims.jsonl"
    needed_doc_ids_out = target_dir / "needed_doc_ids.json"
    schema_report_out = target_dir / "schema_report.json"

    all_doc_ids = set()
    field_types: Dict[str, str] = {}

    with claims_out.open("w", encoding="utf-8") as fout:
        for row in ds:
            # keep raw row exactly as downloaded
            fout.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

            for k, v in row.items():
                if k not in field_types:
                    field_types[k] = type(v).__name__

            evidence = row.get("evidence", {})
            if isinstance(evidence, dict):
                for doc_id in evidence.keys():
                    all_doc_ids.add(str(doc_id))

    with needed_doc_ids_out.open("w", encoding="utf-8") as f:
        json.dump(sorted(all_doc_ids), f, indent=2)

    # save light schema report
    first_rows: List[dict] = [dict(ds[i]) for i in range(min(2, len(ds)))]
    schema_report = {
        "dataset": "davidheineman/scifact-open",
        "split": "claims",
        "num_claims": len(ds),
        "top_level_fields": list(ds.features.keys()) if hasattr(ds, "features") else list(first_rows[0].keys()),
        "field_types": field_types,
        "num_unique_doc_ids": len(all_doc_ids),
        "sample_rows": first_rows,
    }

    with schema_report_out.open("w", encoding="utf-8") as f:
        json.dump(schema_report, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Wrote claims to: {claims_out}")
    print(f"Wrote needed doc ids to: {needed_doc_ids_out}")
    print(f"Wrote schema report to: {schema_report_out}")
    print(f"Claims: {len(ds)}")
    print(f"Unique doc ids referenced: {len(all_doc_ids)}")


if __name__ == "__main__":
    main()