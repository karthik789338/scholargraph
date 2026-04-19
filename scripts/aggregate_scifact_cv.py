from pathlib import Path
import json
import statistics as stats

base = Path("data/processed/cv")
summaries = sorted(base.glob("*/summary.json"))

if not summaries:
    raise SystemExit("No fold summaries found under data/processed/cv/*/summary.json")

rows = [json.load(open(p, "r", encoding="utf-8")) for p in summaries]

def mean_std(key):
    vals = [r[key] for r in rows]
    mean = stats.mean(vals)
    std = stats.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std

keys = [
    "flat_dev_macro_f1",
    "flat_dev_evidence_micro_f1",
    "graph_dev_macro_f1",
    "graph_dev_evidence_micro_f1",
    "delta_macro_f1",
    "delta_evidence_micro_f1",
]

out = {"num_folds": len(rows), "folds": rows, "aggregate": {}}
for key in keys:
    mean, std = mean_std(key)
    out["aggregate"][key] = {"mean": mean, "std": std}

print(json.dumps(out, indent=2))
