from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines.scifact_llm_utils import build_prompt_candidates
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


def truncate_text(text: str, max_chars: int = 900) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_global_user_prompt(claim_text: str, prompt_candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("You are given a scientific claim and retrieved evidence chunks from scientific papers.")
    lines.append("Decide whether the evidence SUPPORTS, REFUTES, or is INSUFFICIENT for the claim.")
    lines.append("Return only one label: SUPPORTS or REFUTES or INSUFFICIENT.")
    lines.append("")
    lines.append(f"Claim: {claim_text}")
    lines.append("")
    lines.append("Evidence chunks:")
    if not prompt_candidates:
        lines.append("[1] No evidence available.")
    else:
        for i, chunk in enumerate(prompt_candidates, start=1):
            title = chunk.get("title", "")
            section = chunk.get("section_title", "")
            text = truncate_text(str(chunk.get("text", "")), max_chars=900)
            prefix_bits = []
            if title:
                prefix_bits.append(f"Title: {title}")
            if section:
                prefix_bits.append(f"Section: {section}")
            prefix = " | ".join(prefix_bits)
            if prefix:
                lines.append(f"[{i}] {prefix} | {text}")
            else:
                lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def build_single_chunk_user_prompt(claim_text: str, chunk: Dict[str, Any]) -> str:
    text = truncate_text(str(chunk.get("text", "")), max_chars=900)
    return (
        "You are given a scientific claim and one evidence chunk from a scientific paper.\n"
        "Decide whether this chunk SUPPORTS, REFUTES, or is INSUFFICIENT for the claim.\n"
        "Return only one label: SUPPORTS or REFUTES or INSUFFICIENT.\n\n"
        f"Claim: {claim_text}\n\n"
        f"Evidence chunk:\n{text}"
    )


def render_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful scientific fact-checking assistant. Answer with exactly one label.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "device_map": "auto" if device.type == "cuda" else None,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    return tokenizer, model


def score_candidate_responses(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    candidate_texts: List[str],
    max_input_length: int,
) -> Dict[str, float]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    target_ids_list = [
        tokenizer(text, add_special_tokens=False)["input_ids"] for text in candidate_texts
    ]

    max_target_len = max(len(t) for t in target_ids_list)
    max_prompt_len = max_input_length - max_target_len - 1
    if max_prompt_len < 1:
        max_prompt_len = 1

    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_tensors = []
    label_tensors = []

    for target_ids in target_ids_list:
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))

    pad_id = tokenizer.pad_token_id
    max_len = max(t.size(0) for t in input_tensors)

    padded_inputs = []
    padded_labels = []
    attention_masks = []

    for inp, lab in zip(input_tensors, label_tensors):
        pad_len = max_len - inp.size(0)
        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0)
        )
        padded_labels.append(
            torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
        )
        attention_masks.append(
            torch.cat([torch.ones(inp.size(0), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
        )

    input_ids = torch.stack(padded_inputs).to(device)
    labels = torch.stack(padded_labels).to(device)
    attention_mask = torch.stack(attention_masks).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())

    valid_mask = (shift_labels != -100).float()
    seq_logprobs = -(token_losses * valid_mask).sum(dim=1)

    return {
        candidate_text: float(score.item())
        for candidate_text, score in zip(candidate_texts, seq_logprobs)
    }


def predict_label(
    model,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    max_input_length: int,
) -> Tuple[str, Dict[str, float]]:
    label_scores = score_candidate_responses(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_text=prompt_text,
        candidate_texts=list(LABEL_TEXTS.values()),
        max_input_length=max_input_length,
    )

    mapped_scores = {
        label_name: label_scores[label_text]
        for label_name, label_text in LABEL_TEXTS.items()
    }
    pred = max(mapped_scores.items(), key=lambda x: x[1])[0]
    return pred, mapped_scores


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
        user_prompt = build_single_chunk_user_prompt(claim_text=claim_text, chunk=candidate)
        prompt_text = render_chat_prompt(tokenizer, user_prompt)

        local_label, local_scores = predict_label(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=prompt_text,
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

    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=3072)
    parser.add_argument("--max-selected-chunks", type=int, default=1)

    args = parser.parse_args()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Running SciFact Llama label-scoring baseline on device=%s with model=%s", device, args.model_name)

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

        user_prompt = build_global_user_prompt(claim_text=query.text, prompt_candidates=prompt_candidates)
        prompt_text = render_chat_prompt(tokenizer, user_prompt)

        predicted_label, label_scores = predict_label(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_text=prompt_text,
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
