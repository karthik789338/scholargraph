from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

def main():
    target_dir = Path("data/raw/scifact")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive_path = tmpdir / "scifact_data.tar.gz"

        print("Downloading SciFact...")
        urllib.request.urlretrieve(SCIFACT_URL, archive_path)

        print("Extracting archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        extracted_data_dir = tmpdir / "data"
        if not extracted_data_dir.exists():
            raise FileNotFoundError("Expected extracted folder 'data/' was not found.")

        needed_files = [
            "corpus.jsonl",
            "claims_train.jsonl",
            "claims_dev.jsonl",
            "claims_test.jsonl",
        ]

        for filename in needed_files:
            src = extracted_data_dir / filename
            if src.exists():
                dst = target_dir / filename
                shutil.copy2(src, dst)
                print(f"Copied: {src} -> {dst}")
            else:
                print(f"Warning: {filename} not found in archive")

    print("\nDone. Files now in:")
    for p in sorted(target_dir.glob("*")):
        print(" -", p)

if __name__ == "__main__":
    main()