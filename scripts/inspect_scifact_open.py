from __future__ import annotations

import json
from pathlib import Path


def main():
    base = Path("data/raw/scifact_open")
    schema_path = base / "schema_report.json"
    claims_path = base / "claims.jsonl"
    doc_ids_path = base / "needed_doc_ids.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema report: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    print("SciFact-Open schema summary")
    print("-" * 40)
    print("Claims:", schema["num_claims"])
    print("Fields:", schema["top_level_fields"])
    print("Field types:", schema["field_types"])
    print("Unique doc ids referenced:", schema["num_unique_doc_ids"])
    print("\nFirst sample row:")
    print(json.dumps(schema["sample_rows"][0], indent=2)[:4000])

    if doc_ids_path.exists():
        doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        print("\nFirst 20 needed doc ids:")
        print(doc_ids[:20])


if __name__ == "__main__":
    main()