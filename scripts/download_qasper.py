from pathlib import Path
import json
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FILES = {
    "train": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet",
}


def main():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Please install required packages first: pip install datasets pyarrow"
        ) from e

    target_dir = Path("data/raw/qasper")
    target_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = target_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in FILES.items():
        parquet_path = parquet_dir / f"{split_name}.parquet"

        print(f"Downloading {split_name} parquet...")
        urllib.request.urlretrieve(url, parquet_path)

        print(f"Loading {split_name} split from local parquet...")
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")

        out = {}
        for idx, ex in enumerate(ds):
            paper_id = str(ex.get("id") or ex.get("paper_id") or idx)
            out[paper_id] = dict(ex)

        out_path = target_dir / f"{split_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

        print(f"Wrote {len(out)} papers to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()