from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_claim_id(query_obj: Dict[str, Any]) -> int:
    # Try explicit fields first
    for key in ["id", "claim_id"]:
        if key in query_obj:
            return int(query_obj[key])

    # Fallback to query_id like scifact_query_123
    query_id = str(query_obj.get("query_id", ""))
    m = re.search(r"(\d+)$", query_id)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not extract claim id from query object keys={list(query_obj.keys())}")


def normalize_label(label: str) -> Optional[str]:
    label = str(label).strip().lower()
    if label == "supports":
        return "SUPPORT"
    if label == "refutes":
        return "CONTRADICT"
    if label == "insufficient":
        return None
    return None


def extract_doc_id_from_chunk(chunk_obj: Dict[str, Any]) -> str:
    # Try direct fields
    for key in ["doc_id", "document_id", "paper_id"]:
        if key in chunk_obj and chunk_obj[key] is not None:
            return str(chunk_obj[key])

    # metadata fallback
    metadata = chunk_obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ["doc_id", "document_id", "paper_id"]:
            if key in metadata and metadata[key] is not None:
                return str(metadata[key])

    # Parse from chunk_id like scifact_doc_13734012_...
    chunk_id = str(chunk_obj.get("chunk_id", ""))
    m = re.search(r"scifact_doc_(\d+)", chunk_id)
    if m:
        return m.group(1)

    raise ValueError(f"Could not extract doc id from chunk: {chunk_obj.get('chunk_id')}")


def extract_sentence_indices_from_chunk(chunk_obj: Dict[str, Any]) -> List[int]:
    candidates = []

    for key in ["sentence_indices", "sent_indices", "sentences", "sentence_ids"]:
        if key in chunk_obj and chunk_obj[key] is not None:
            candidates = chunk_obj[key]
            break

    if not candidates:
        metadata = chunk_obj.get("metadata")
        if isinstance(metadata, dict):
            for key in ["sentence_indices", "sent_indices", "sentences", "sentence_ids"]:
                if key in metadata and metadata[key] is not None:
                    candidates = metadata[key]
                    break

    out: List[int] = []

    if isinstance(candidates, list):
        for x in candidates:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str):
                m = re.search(r"_sent_(\d+)$", x)
                if m:
                    out.append(int(m.group(1)))
                elif x.isdigit():
                    out.append(int(x))

    out = sorted(set(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries = read_jsonl(args.queries)
    predictions = read_jsonl(args.predictions)
    chunks = read_jsonl(args.chunks)

    query_by_qid = {str(q["query_id"]): q for q in queries}
    chunk_by_id = {str(c["chunk_id"]): c for c in chunks}

    official_rows: List[Dict[str, Any]] = []

    for pred in predictions:
        query_id = str(pred["query_id"])
        query_obj = query_by_qid[query_id]
        claim_id = extract_claim_id(query_obj)

        pred_label = normalize_label(pred.get("predicted_label", ""))
        pred_chunks = pred.get("predicted_evidence_chunks", []) or []

        evidence: Dict[str, Dict[str, Any]] = {}

        if pred_label is not None:
            by_doc: Dict[str, List[int]] = {}
            for chunk_id in pred_chunks:
                chunk_obj = chunk_by_id.get(str(chunk_id))
                if chunk_obj is None:
                    continue
                doc_id = extract_doc_id_from_chunk(chunk_obj)
                sent_ids = extract_sentence_indices_from_chunk(chunk_obj)
                by_doc.setdefault(doc_id, [])
                by_doc[doc_id].extend(sent_ids)

            for doc_id, sent_ids in by_doc.items():
                sent_ids = sorted(set(int(x) for x in sent_ids))
                evidence[str(doc_id)] = {
                    "label": pred_label,
                    "sentences": sent_ids,
                }

        official_rows.append(
            {
                "id": claim_id,
                "evidence": evidence,
            }
        )

    write_jsonl(args.output, official_rows)
    print(f"Wrote {len(official_rows)} official-format predictions to {args.output}")


if __name__ == "__main__":
    main()
