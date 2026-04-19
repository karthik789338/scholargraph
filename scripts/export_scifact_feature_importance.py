from pathlib import Path
import json
from collections import defaultdict

base = Path("data/processed/cv")
fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold")])

agg = {
    "supports": defaultdict(list),
    "refutes": defaultdict(list),
    "insufficient": defaultdict(list),
}

for fold_dir in fold_dirs:
    path = fold_dir / "graph_feature_classifier_compact" / "feature_importance.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for label, rows in data.items():
        for row in rows:
            agg[label][row["feature"]].append(row["weight"])

summary = {}
for label, feat_map in agg.items():
    rows = []
    for feat, weights in feat_map.items():
        mean_weight = sum(weights) / len(weights)
        mean_abs_weight = sum(abs(w) for w in weights) / len(weights)
        rows.append({
            "feature": feat,
            "mean_weight": mean_weight,
            "mean_abs_weight": mean_abs_weight,
            "num_folds": len(weights),
        })
    rows = sorted(rows, key=lambda x: x["mean_abs_weight"], reverse=True)
    summary[label] = rows

out_dir = Path("reports/scifact_frozen")
out_dir.mkdir(parents=True, exist_ok=True)

with (out_dir / "feature_importance_aggregated.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

md = []
md.append("# Aggregated Feature Importance")
md.append("")
for label in ["supports", "refutes", "insufficient"]:
    md.append(f"## {label}")
    md.append("")
    md.append("| Feature | Mean weight | Mean abs weight | Folds |")
    md.append("|---|---:|---:|---:|")
    for row in summary[label][:15]:
        md.append(
            f"| {row['feature']} | {row['mean_weight']:.4f} | "
            f"{row['mean_abs_weight']:.4f} | {row['num_folds']} |"
        )
    md.append("")

(out_dir / "feature_importance_aggregated.md").write_text("\n".join(md), encoding="utf-8")

print(f"Wrote:\n- {out_dir/'feature_importance_aggregated.json'}\n- {out_dir/'feature_importance_aggregated.md'}")
