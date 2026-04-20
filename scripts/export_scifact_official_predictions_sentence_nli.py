from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    for key in ["id", "claim_id"]:
        if key in query_obj:
            return int(query_obj[key])

    query_id = str(query_obj.get("query_id", ""))
    m = re.search(r"(\d+)$", query_id)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not extract claim id from query object: {query_obj}")


def normalize_pred_label(label: str) -> Optional[str]:
    label = str(label).strip().lower()
    if label == "supports":
        return "SUPPORT"
    if label == "refutes":
        return "CONTRADICT"
    if label == "insufficient":
        return None
    return None


def extract_doc_id_from_chunk(chunk_obj: Dict[str, Any]) -> str:
    for key in ["doc_id", "document_id", "paper_id"]:
        if key in chunk_obj and chunk_obj[key] is not None:
            val = str(chunk_obj[key])
            return val.replace("scifact_doc_", "")

    metadata = chunk_obj.get("metadata")
    if isinstance(metadata, dict):
        for key in ["doc_id", "document_id", "paper_id"]:
            if key in metadata and metadata[key] is not None:
                val = str(metadata[key])
                return val.replace("scifact_doc_", "")

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

    return sorted(set(out))


def load_corpus_sentences(path: str) -> Dict[str, List[str]]:
    rows = read_jsonl(path)
    out: Dict[str, List[str]] = {}

    for row in rows:
        doc_id = None
        for key in ["doc_id", "document_id", "paper_id", "id"]:
            if key in row and row[key] is not None:
                doc_id = str(row[key]).replace("scifact_doc_", "")
                break

        if doc_id is None:
            continue

        # SciFact corpus commonly stores abstract as sentence list
        if isinstance(row.get("abstract"), list):
            out[doc_id] = [str(x) for x in row["abstract"]]
        elif isinstance(row.get("sentences"), list):
            out[doc_id] = [str(x) for x in row["sentences"]]
        elif isinstance(row.get("text"), list):
            out[doc_id] = [str(x) for x in row["text"]]
        else:
            out[doc_id] = []

    return out


def load_nli_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs: Dict[str, Any] = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    label_map = {}
    for idx, raw_label in id2label.items():
        lab = str(raw_label).lower()
        if "entail" in lab or "support" in lab:
            label_map["SUPPORT"] = int(idx)
        elif "contrad" in lab or "refut" in lab:
            label_map["CONTRADICT"] = int(idx)
        elif "neutral" in lab or "insuff" in lab:
            label_map["NEUTRAL"] = int(idx)

    if "SUPPORT" not in label_map or "CONTRADICT" not in label_map:
        raise ValueError(f"Could not build label map from id2label={id2label}")

    return tokenizer, model, label_map


def score_sentences(
    claim: str,
    sentences: List[str],
    tokenizer,
    model,
    label_map: Dict[str, int],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[Dict[str, float]]:
    outputs_all: List[Dict[str, float]] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            [claim] * len(batch),
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).detach().cpu()

        for row in probs:
            result = {
                "SUPPORT": float(row[label_map["SUPPORT"]]),
                "CONTRADICT": float(row[label_map["CONTRADICT"]]),
                "NEUTRAL": float(row[label_map["NEUTRAL"]]) if "NEUTRAL" in label_map else 0.0,
            }
            outputs_all.append(result)

    return outputs_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--model-name", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-sentences-per-doc", type=int, default=1)

    args = parser.parse_args()

    queries = read_jsonl(args.queries)
    predictions = read_jsonl(args.predictions)
    chunks = read_jsonl(args.chunks)
    corpus_sentences = load_corpus_sentences(args.corpus)

    query_by_qid = {str(q["query_id"]): q for q in queries}
    chunk_by_id = {str(c["chunk_id"]): c for c in chunks}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, label_map = load_nli_model(args.model_name, device)

    official_rows: List[Dict[str, Any]] = []

    for idx, pred in enumerate(predictions, start=1):
        query_id = str(pred["query_id"])
        query_obj = query_by_qid[query_id]
        claim_id = extract_claim_id(query_obj)
        claim_text = query_obj["text"]

        pred_label = normalize_pred_label(pred.get("predicted_label", ""))
        pred_chunks = pred.get("predicted_evidence_chunks", []) or []

        evidence: Dict[str, Dict[str, Any]] = {}

        if pred_label is not None:
            by_doc_sentence_candidates: Dict[str, List[Tuple[int, str]]] = {}

            for chunk_id in pred_chunks:
                chunk_obj = chunk_by_id.get(str(chunk_id))
                if chunk_obj is None:
                    continue

                doc_id = extract_doc_id_from_chunk(chunk_obj)
                sent_indices = extract_sentence_indices_from_chunk(chunk_obj)
                doc_sents = corpus_sentences.get(doc_id, [])

                for sidx in sent_indices:
                    if 0 <= sidx < len(doc_sents):
                        by_doc_sentence_candidates.setdefault(doc_id, [])
                        by_doc_sentence_candidates[doc_id].append((sidx, doc_sents[sidx]))

            for doc_id, sent_pairs in by_doc_sentence_candidates.items():
                # deduplicate by sentence index
                dedup = {}
                for sidx, stext in sent_pairs:
                    dedup[sidx] = stext
                sent_pairs = sorted(dedup.items(), key=lambda x: x[0])

                sentence_indices = [sidx for sidx, _ in sent_pairs]
                sentence_texts = [stext for _, stext in sent_pairs]

                if not sentence_texts:
                    continue

                scores = score_sentences(
                    claim=claim_text,
                    sentences=sentence_texts,
                    tokenizer=tokenizer,
                    model=model,
                    label_map=label_map,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )

                target_key = "SUPPORT" if pred_label == "SUPPORT" else "CONTRADICT"
                ranked = sorted(
                    zip(sentence_indices, scores),
                    key=lambda x: x[1][target_key],
                    reverse=True,
                )

                chosen = [sidx for sidx, _ in ranked[: args.top_sentences_per_doc]]

                evidence[str(doc_id)] = {
                    "label": pred_label,
                    "sentences": sorted(set(chosen)),
                }

        official_rows.append(
            {
                "id": claim_id,
                "evidence": evidence,
            }
        )

        if idx % 50 == 0 or idx == len(predictions):
            print(f"Processed {idx}/{len(predictions)} predictions")

    write_jsonl(args.output, official_rows)
    print(f"Wrote {len(official_rows)} sentence-NLI official predictions to {args.output}")


if __name__ == "__main__":
    main()
