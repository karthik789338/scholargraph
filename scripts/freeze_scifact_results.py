from pathlib import Path
import json
import statistics as stats

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def mean_std(values):
    return {
        "mean": stats.mean(values),
        "std": stats.pstdev(values) if len(values) > 1 else 0.0,
    }

base = Path("data/processed/cv")
fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold")])

flat_rows = []
graph_rule_rows = []
compact_rows = []

for fold_dir in fold_dirs:
    flat_summary = load_json(fold_dir / "summary.json")
    compact_summary = load_json(fold_dir / "graph_feature_classifier_compact" / "summary.json")

    flat_rows.append({
        "fold": fold_dir.name,
        "macro_f1": flat_summary["flat_dev_macro_f1"],
        "evidence_f1": flat_summary["flat_dev_evidence_micro_f1"],
    })
    graph_rule_rows.append({
        "fold": fold_dir.name,
        "macro_f1": flat_summary["graph_dev_macro_f1"],
        "evidence_f1": flat_summary["graph_dev_evidence_micro_f1"],
    })
    compact_rows.append({
        "fold": fold_dir.name,
        "macro_f1": compact_summary["dev_label_macro_f1"],
        "evidence_f1": compact_summary["dev_evidence_micro_f1"],
    })

frozen_dir = Path("reports/scifact_frozen")
frozen_dir.mkdir(parents=True, exist_ok=True)

report = {
    "num_folds": len(fold_dirs),
    "flat_baseline": {
        "macro_f1": mean_std([r["macro_f1"] for r in flat_rows]),
        "evidence_micro_f1": mean_std([r["evidence_f1"] for r in flat_rows]),
        "folds": flat_rows,
    },
    "graph_rule": {
        "macro_f1": mean_std([r["macro_f1"] for r in graph_rule_rows]),
        "evidence_micro_f1": mean_std([r["evidence_f1"] for r in graph_rule_rows]),
        "folds": graph_rule_rows,
    },
    "graph_feature_compact": {
        "macro_f1": mean_std([r["macro_f1"] for r in compact_rows]),
        "evidence_micro_f1": mean_std([r["evidence_f1"] for r in compact_rows]),
        "folds": compact_rows,
    },
}

with (frozen_dir / "scifact_results_frozen.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

md = []
md.append("# Frozen SciFact Results")
md.append("")
md.append("| Model | Macro F1 (mean±std) | Evidence micro F1 (mean±std) |")
md.append("|---|---:|---:|")
md.append(
    f"| Flat baseline | "
    f"{report['flat_baseline']['macro_f1']['mean']:.4f} ± {report['flat_baseline']['macro_f1']['std']:.4f} | "
    f"{report['flat_baseline']['evidence_micro_f1']['mean']:.4f} ± {report['flat_baseline']['evidence_micro_f1']['std']:.4f} |"
)
md.append(
    f"| Graph rule | "
    f"{report['graph_rule']['macro_f1']['mean']:.4f} ± {report['graph_rule']['macro_f1']['std']:.4f} | "
    f"{report['graph_rule']['evidence_micro_f1']['mean']:.4f} ± {report['graph_rule']['evidence_micro_f1']['std']:.4f} |"
)
md.append(
    f"| Graph-feature compact | "
    f"{report['graph_feature_compact']['macro_f1']['mean']:.4f} ± {report['graph_feature_compact']['macro_f1']['std']:.4f} | "
    f"{report['graph_feature_compact']['evidence_micro_f1']['mean']:.4f} ± {report['graph_feature_compact']['evidence_micro_f1']['std']:.4f} |"
)

(frozen_dir / "scifact_results_frozen.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nWrote:\n- {frozen_dir/'scifact_results_frozen.json'}\n- {frozen_dir/'scifact_results_frozen.md'}")
