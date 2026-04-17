from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


def main():
    out_dir = Path("data/raw/scifact_open/full_corpus")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "corpus.jsonl"
    manifest_path = out_dir / "source_manifest.json"

    print("Downloading full SciFact-Open corpus from HF mirror...")
    ds = load_dataset("if-ir/scifact_open", name="corpus", split="corpus")

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds, start=1):
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            if i % 50000 == 0:
                print(f"Wrote {i} rows...")

    sample = ds[0]
    manifest = {
        "source": "if-ir/scifact_open",
        "subset": "corpus",
        "split": "corpus",
        "num_rows": len(ds),
        "sample_keys": list(sample.keys()),
        "local_path": str(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Wrote corpus to: {out_path}")
    print(f"Wrote manifest to: {manifest_path}")
    print(f"Rows: {len(ds)}")


if __name__ == "__main__":
    main()