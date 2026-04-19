from pathlib import Path
import json
from scipy.stats import ttest_rel

base = Path("data/processed/cv")
fold_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("fold")])

flat_macro = []
flat_evidence = []
graph_macro = []
graph_evidence = []
compact_macro = []
compact_evidence = []

for fold_dir in fold_dirs:
    s = json.load(open(fold_dir / "summary.json", encoding="utf-8"))
    c = json.load(open(fold_dir / "graph_feature_classifier_compact" / "summary.json", encoding="utf-8"))

    flat_macro.append(s["flat_dev_macro_f1"])
    flat_evidence.append(s["flat_dev_evidence_micro_f1"])
    graph_macro.append(s["graph_dev_macro_f1"])
    graph_evidence.append(s["graph_dev_evidence_micro_f1"])
    compact_macro.append(c["dev_label_macro_f1"])
    compact_evidence.append(c["dev_evidence_micro_f1"])

out = {
    "compact_vs_flat_macro": {
        "t_stat": float(ttest_rel(compact_macro, flat_macro).statistic),
        "p_value": float(ttest_rel(compact_macro, flat_macro).pvalue),
    },
    "compact_vs_flat_evidence": {
        "t_stat": float(ttest_rel(compact_evidence, flat_evidence).statistic),
        "p_value": float(ttest_rel(compact_evidence, flat_evidence).pvalue),
    },
    "compact_vs_graph_rule_macro": {
        "t_stat": float(ttest_rel(compact_macro, graph_macro).statistic),
        "p_value": float(ttest_rel(compact_macro, graph_macro).pvalue),
    },
    "compact_vs_graph_rule_evidence": {
        "t_stat": float(ttest_rel(compact_evidence, graph_evidence).statistic),
        "p_value": float(ttest_rel(compact_evidence, graph_evidence).pvalue),
    },
}

out_dir = Path("reports/scifact_frozen")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "scifact_significance.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
