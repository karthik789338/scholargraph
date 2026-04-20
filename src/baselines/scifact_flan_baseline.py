from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.baselines.scifact_llm_utils import (
    build_prompt_candidates,
    build_scifact_prompt,
    parse_scifact_output,
)
from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.scifact_graph_verdict import load_queries


LOGGER = logging.getLogger(__name__)


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
            if not line:
                continue
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


def generate_batch(
    prompts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--graph-inputs", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--model-name", default="google/flan-t5-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-chunk-chars", type=int, default=900)
    parser.add_argument("--default-top1-evidence-on-empty", action="store_true")

    args = parser.parse_args()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running SciFact FLAN baseline on device=%s with model=%s", device, args.model_name)

    queries = load_queries(args.queries)
    graph_inputs = read_jsonl(args.graph_inputs)
    chunks = read_jsonl(args.chunks)

    graph_inputs_by_qid = map_by_id(graph_inputs, "query_id")
    chunks_by_id = map_by_id(chunks, "chunk_id")

    tokenizer, model = load_model_and_tokenizer(args.model_name, device)

    prompt_rows: List[Dict[str, Any]] = []
    for query in queries:
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
        prompt_rows.append(
            {
                "query": query,
                "prompt_candidates": prompt_candidates,
                "prompt": prompt,
            }
        )

    LOGGER.info("Prepared %d prompts", len(prompt_rows))

    predictions: List[Dict[str, Any]] = []
    parse_failures = 0
    empty_evidence_non_insufficient = 0

    for start in range(0, len(prompt_rows), args.batch_size):
        batch_rows = prompt_rows[start : start + args.batch_size]
        batch_prompts = [row["prompt"] for row in batch_rows]
        batch_outputs = generate_batch(
            prompts=batch_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )

        for row, raw_output in zip(batch_rows, batch_outputs):
            query = row["query"]
            prompt_candidates = row["prompt_candidates"]

            parsed = parse_scifact_output(
                raw_output=raw_output,
                num_candidates=len(prompt_candidates),
            )

            predicted_label = parsed["predicted_label"]
            predicted_chunk_numbers = parsed["predicted_chunk_numbers"]

            if predicted_label is None:
                parse_failures += 1
                predicted_label = "insufficient"
                predicted_chunk_numbers = []

            predicted_evidence_chunks: List[str] = []
            for n in predicted_chunk_numbers:
                idx = n - 1
                if 0 <= idx < len(prompt_candidates):
                    predicted_evidence_chunks.append(prompt_candidates[idx]["chunk_id"])

            if predicted_label != "insufficient" and not predicted_evidence_chunks:
                empty_evidence_non_insufficient += 1
                if args.default_top1_evidence_on_empty and prompt_candidates:
                    predicted_evidence_chunks = [prompt_candidates[0]["chunk_id"]]

            predictions.append(
                {
                    "query_id": query.query_id,
                    "predicted_label": predicted_label,
                    "predicted_evidence_chunks": predicted_evidence_chunks,
                    "raw_output": raw_output,
                    "prompt_chunk_ids": [c["chunk_id"] for c in prompt_candidates],
                }
            )

        LOGGER.info("Processed %d/%d queries", min(start + args.batch_size, len(prompt_rows)), len(prompt_rows))

    metrics = evaluate_scifact_predictions(
        queries=queries,
        predictions=predictions,
    )
    metrics["generation_stats"] = {
        "num_examples": len(queries),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / max(1, len(queries)),
        "empty_evidence_non_insufficient": empty_evidence_non_insufficient,
        "model_name": args.model_name,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
    }

    write_jsonl(args.output_predictions, predictions)
    write_json(args.output_metrics, metrics)

    LOGGER.info("Wrote predictions to %s", args.output_predictions)
    LOGGER.info("Wrote metrics to %s", args.output_metrics)
    LOGGER.info(
        "Label macro F1: %.4f | Evidence micro F1: %.4f | Parse failure rate: %.4f",
        metrics["label_metrics"]["macro_f1"],
        metrics["evidence_metrics"]["micro_f1"],
        metrics["generation_stats"]["parse_failure_rate"],
    )


if __name__ == "__main__":
    main()
