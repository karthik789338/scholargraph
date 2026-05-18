from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_md(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = obj["summary_rows"]
    headers = list(rows[0].keys())

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Key derived quantities")
    for k, v in obj["derived"].items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def stage_drop(a: float, b: float) -> float:
    return float(a - b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose internal vs official SciFact gaps.")
    parser.add_argument("--flat-internal", required=True, help="Flat internal metrics JSON")
    parser.add_argument("--compact-internal", required=True, help="Compact internal metrics JSON")
    parser.add_argument("--flat-official", required=True, help="Flat official metrics JSON")
    parser.add_argument("--compact-official", required=True, help="Compact official metrics JSON")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    flat_internal = read_json(args.flat_internal)
    compact_internal = read_json(args.compact_internal)
    flat_official = read_json(args.flat_official)
    compact_official = read_json(args.compact_official)

    fi_macro = float(flat_internal["label_metrics"]["macro_f1"])
    fi_evid = float(flat_internal["evidence_metrics"]["micro_f1"])

    ci_macro = float(compact_internal["label_metrics"]["macro_f1"])
    ci_evid = float(compact_internal["evidence_metrics"]["micro_f1"])

    fo_label_only = float(flat_official["abstract_label_only"]["f1"])
    fo_label_rat = float(flat_official["abstract_label_rationale"]["f1"])
    fo_sent = float(flat_official["sentence_selection_label"]["f1"])

    co_label_only = float(compact_official["abstract_label_only"]["f1"])
    co_label_rat = float(compact_official["abstract_label_rationale"]["f1"])
    co_sent = float(compact_official["sentence_selection_label"]["f1"])

    summary_rows = [
        {
            "system": "flat",
            "internal_macro_f1": fi_macro,
            "internal_evidence_micro_f1": fi_evid,
            "official_abstract_label_only_f1": fo_label_only,
            "official_abstract_label_rationale_f1": fo_label_rat,
            "official_sentence_selection_label_f1": fo_sent,
            "drop_labelonly_to_rationale": stage_drop(fo_label_only, fo_label_rat),
            "drop_rationale_to_sentence": stage_drop(fo_label_rat, fo_sent),
            "projection_gap_internal_evidence_to_official_sentence": stage_drop(fi_evid, fo_sent),
        },
        {
            "system": "compact",
            "internal_macro_f1": ci_macro,
            "internal_evidence_micro_f1": ci_evid,
            "official_abstract_label_only_f1": co_label_only,
            "official_abstract_label_rationale_f1": co_label_rat,
            "official_sentence_selection_label_f1": co_sent,
            "drop_labelonly_to_rationale": stage_drop(co_label_only, co_label_rat),
            "drop_rationale_to_sentence": stage_drop(co_label_rat, co_sent),
            "projection_gap_internal_evidence_to_official_sentence": stage_drop(ci_evid, co_sent),
        },
    ]

    derived = {
        "compact_minus_flat_internal_macro_f1": ci_macro - fi_macro,
        "compact_minus_flat_internal_evidence_micro_f1": ci_evid - fi_evid,
        "compact_minus_flat_official_abstract_label_only_f1": co_label_only - fo_label_only,
        "compact_minus_flat_official_abstract_label_rationale_f1": co_label_rat - fo_label_rat,
        "compact_minus_flat_official_sentence_selection_label_f1": co_sent - fo_sent,
        "flat_counts": flat_official.get("counts", {}),
        "compact_counts": compact_official.get("counts", {}),
    }

    payload = {
        "summary_rows": summary_rows,
        "derived": derived,
        "inputs": {
            "flat_internal": args.flat_internal,
            "compact_internal": args.compact_internal,
            "flat_official": args.flat_official,
            "compact_official": args.compact_official,
        },
    }

    write_json(args.output_json, payload)
    write_md(args.output_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
