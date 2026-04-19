from pathlib import Path
import json
import random
import argparse


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create 5-fold SciFact CV splits from claims_train.jsonl")
    parser.add_argument(
        "--input",
        default="data/raw/scifact/claims_train.jsonl",
        help="Path to SciFact train claims JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/scifact/cross_validation",
        help="Directory to write fold_{i}/claims_train.jsonl and claims_dev.jsonl",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input claims file not found: {input_path}")

    claims = read_jsonl(input_path)
    if not claims:
        raise ValueError(f"No claims found in {input_path}")

    random.seed(args.seed)
    claims = claims[:]  # copy
    random.shuffle(claims)

    n = len(claims)
    k = args.num_folds
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        end = start + size
        folds.append(claims[start:end])
        start = end

    for i in range(k):
        dev_claims = folds[i]
        train_claims = []
        for j in range(k):
            if j != i:
                train_claims.extend(folds[j])

        fold_dir = output_dir / f"fold_{i}"
        write_jsonl(train_claims, fold_dir / "claims_train.jsonl")
        write_jsonl(dev_claims, fold_dir / "claims_dev.jsonl")

        print(
            f"fold_{i}: train={len(train_claims)} dev={len(dev_claims)} "
            f"-> {fold_dir}"
        )


if __name__ == "__main__":
    main()
