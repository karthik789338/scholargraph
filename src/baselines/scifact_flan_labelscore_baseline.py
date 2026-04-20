from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.baselines.scifact_llm_utils import build_prompt_candidates, build_scifact_prompt
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.scifact_graph_verdict import load_queries


LOGGER = logging.getLogger(__name__)

LABEL_TEXTS = {
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "insufficient": "INSUFFICIENT",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def map_by_id(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if key in row:
            out[str(row[key])] = row
    return out


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs: Dict[str, Any] = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def score_target_text(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    target_text: str,
    max_input_length: int,
) -> float:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    labels = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    label_ids = labels["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(**enc, labels=label_ids)

    # HF loss is mean over target tokens, so recover sequence score
    token_count = int(label_ids.shape[1])
    seq_logprob = -float(outputs.loss.item()) * token_count
    return seq_logprob


def predict_label(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_input_length: int,
) -> Tuple[str, Dict[str, float]]:
    scores = {}
    for label_name, label_text in LABEL_TEXTS.items():
        scores[label_name] = score_target_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            target_text=label_text,
            max_input_length=max_input_length,
        )
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores


def build_single_chunk_prompt(claim_text: str, chunk: Dict[str, Any], max_chunk_chars: int = 900) -> str:
    chunk_text = str(chunk.get("text", ""))
    if len(chunk_text) > max_chunk_chars:
        chunk_text = chunk_text[: max_chunk_chars - 3].rstrip() + "..."

    return (
        "You are given a scientific claim and one retrieved evidence chunk from a scientific paper.\n\n"
        "Task:\n"
        "Decide whether this evidence chunk SUPPORTS, REFUTES, or is INSUFFICIENT for the claim.\n\n"
        f"Claim: {claim_text}\n\n"
        f"Evidence chunk:\n{chunk_text}\n\n"
        "Answer with one label only:\n"
        "SUPPORTS or REFUTES or INSUFFICIENT"
    )


def select_evidence_chunks(
    model,
    tokenizer,
    device: torch.device,
    claim_text: str,
    prompt_candidates: List[Dict[str, Any]],
    global_label: str,
    max_input_length: int,
    max_selected_chunks: int = 1,
) -> List[str]:
    if global_label == "insufficient":
        return []

    scored = []
    for candidate in prompt_candidates:
        prompt = build_single_chunk_prompt(claim_text=claim_text, chunk=candidate)
        local_label, local_scores = predict_label(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_input_length=max_input_length,
        )
        if local_label == global_label:
            margin = local_scores[global_label] - max(
                score for lbl, score in local_scores.items() if lbl != global_label
            )
            scored.append((candidate["chunk_id"], margin))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in scored[:max_selected_chunks]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--graph-inputs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--model-name", default="google/flan-t5-large")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-selected-chunks", type=int, default=1)
    parser.add_argument("--max-chunk-chars", type=int, default=900)

    args = parser.parse_args()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running SciFact FLAN label-scoring baseline on device=%s with model=%s", device, args.model_name)

    queries = load_queries(args.queries)
    graph_inputs = read_jsonl(args.graph_inputs)
    chunks = read_jsonl(args.chunks)

    graph_inputs_by_qid = map_by_id(graph_inputs, "query_id")
    chunks_by_id = map_by_id(chunks, "chunk_id")

    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    predictions: List[Dict[str, Any]] = []
    label_counts = {"supports": 0, "refutes": 0, "insufficient": 0}

    for i, query in enumerate(queries, start=1):
        query_id = str(query.query_id)
        graph_input = graph_inputs_by_qid.get(query_id, {"candidate_chunks": []})

        prompt_candidates = build_prompt_candidates(
            graph_input=graph_input,
            chunks_by_id=chunks_by_id,
            top_k=args.top_k,
        )

        prompt = build_scifact_prompt(
            claim_text=query.text,
            candidate_chunks=prompt_candidates,
            max_chunk_chars=args.max_chunk_chars,
        )

        predicted_label, label_scores = predict_label(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_input_length=args.max_input_length,
        )

        predicted_evidence_chunks = select_evidence_chunks(
            model=model,
            tokenizer=tokenizer,
            device=device,
            claim_text=query.text,
            prompt_candidates=prompt_candidates,
            global_label=predicted_label,
            max_input_length=args.max_input_length,
            max_selected_chunks=args.max_selected_chunks,
        )

        label_counts[predicted_label] += 1

        predictions.append(
            {
                "query_id": query.query_id,
                "predicted_label": predicted_label,
                "predicted_evidence_chunks": predicted_evidence_chunks,
                "label_scores": label_scores,
                "prompt_chunk_ids": [c["chunk_id"] for c in prompt_candidates],
            }
        )

        if i % 25 == 0 or i == len(queries):
            LOGGER.info("Processed %d/%d queries", i, len(queries))

    metrics = evaluate_scifact_predictions(
        queries=queries,
        predictions=predictions,
    )
    metrics["generation_stats"] = {
        "num_examples": len(queries),
        "model_name": args.model_name,
        "top_k": args.top_k,
        "max_input_length": args.max_input_length,
        "max_selected_chunks": args.max_selected_chunks,
        "label_counts": label_counts,
    }

    write_jsonl(args.output_predictions, predictions)
    write_json(args.output_metrics, metrics)

    LOGGER.info("Wrote predictions to %s", args.output_predictions)
    LOGGER.info("Wrote metrics to %s", args.output_metrics)
    LOGGER.info(
        "Label macro F1: %.4f | Evidence micro F1: %.4f",
        metrics["label_metrics"]["macro_f1"],
        metrics["evidence_metrics"]["micro_f1"],
    )


if __name__ == "__main__":
    main()
