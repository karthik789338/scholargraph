from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.qasper_answer_metrics import evaluate_qasper_answer_predictions
from src.graph.schemas import Document, Query
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_documents(path: str | Path) -> List[Document]:
    return [Document(**x) for x in read_jsonl(path)]


def map_documents_by_id(documents: Sequence[Document]) -> Dict[str, Document]:
    return {doc.doc_id: doc for doc in documents}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_seq2seq_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(get_device())
    model.eval()

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def shorten_answer(text: str, max_words: int = 20) -> str:
    text = normalize_text(text)

    for sep in [".", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).strip()

    return text if text else "unanswerable"


def build_full_paper_context(doc: Document) -> str:
    parts: List[str] = []

    if normalize_text(doc.title):
        parts.append(f"Title: {normalize_text(doc.title)}")

    if normalize_text(doc.abstract):
        parts.append(f"Abstract: {normalize_text(doc.abstract)}")

    for sec in doc.sections:
        sec_title = normalize_text(sec.section_title)
        sec_text = normalize_text(sec.section_text)

        if sec_title and sec_text:
            parts.append(f"Section: {sec_title}\n{sec_text}")
        elif sec_text:
            parts.append(sec_text)

    return "\n\n".join(parts).strip()


def build_prompt(question: str, doc: Document) -> str:
    context = build_full_paper_context(doc)

    prompt = (
        "You are answering a question about a scientific paper.\n"
        "Use only the paper content below.\n"
        "If the paper does not support an answer, return exactly: unanswerable.\n"
        "Return a short answer phrase or sentence.\n\n"
        f"Question: {normalize_text(question)}\n\n"
        f"{context}\n\n"
        "Answer:"
    )
    return prompt


def generate_answers(
    prompts: Sequence[str],
    model_name: str,
    batch_size: int = 1,
    max_input_length: int = 4096,
    max_new_tokens: int = 32,
) -> List[str]:
    tokenizer, model = load_seq2seq_model(model_name)
    device = get_device()

    outputs: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])

        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 2,
            "do_sample": False,
            "early_stopping": True,
        }

        # LED benefits from global attention on leading tokens
        if getattr(model.config, "model_type", "") == "led":
            global_attention_mask = torch.zeros_like(inputs["input_ids"])
            global_tokens = min(64, global_attention_mask.shape[1])
            global_attention_mask[:, :global_tokens] = 1
            generation_kwargs["global_attention_mask"] = global_attention_mask.to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = [shorten_answer(x, max_words=20) for x in decoded]
        outputs.extend(decoded)

        end = min(start + batch_size, len(prompts))
        if end % 32 == 0 or end == len(prompts):
            logger.info(f"Generated answers for {end}/{len(prompts)} QASPER queries")

    return outputs


def build_predictions(
    queries: Sequence[Query],
    documents_by_id: Dict[str, Document],
    model_name: str = "allenai/led-base-16384",
    batch_size: int = 1,
    max_input_length: int = 4096,
    max_new_tokens: int = 32,
    max_queries: int | None = None,
) -> List[Dict]:
    working_queries = list(queries)
    if max_queries is not None:
        working_queries = working_queries[:max_queries]

    prompts: List[str] = []
    meta: List[str] = []

    for query in working_queries:
        doc = documents_by_id.get(query.source_doc_id)
        if doc is None:
            prompts.append(
                "You are answering a question about a scientific paper.\n"
                "If the paper does not support an answer, return exactly: unanswerable.\n\n"
                f"Question: {normalize_text(query.text)}\n\n"
                "Answer:"
            )
        else:
            prompts.append(build_prompt(query.text, doc))

        meta.append(query.query_id)

    logger.info(
        f"Generating long-context answers for {len(prompts)} QASPER queries "
        f"with model={model_name}, max_input_length={max_input_length}"
    )

    answers = generate_answers(
        prompts=prompts,
        model_name=model_name,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )

    predictions = []
    for query_id, answer in zip(meta, answers):
        predictions.append(
            {
                "query_id": query_id,
                "predicted_answer": answer,
                "predicted_evidence_chunks": [],
            }
        )

    logger.info(f"Built {len(predictions)} QASPER long-context predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run long-context QASPER baseline")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    documents = load_documents(args.documents)
    documents_by_id = map_documents_by_id(documents)

    predictions = build_predictions(
        queries=queries,
        documents_by_id=documents_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        max_queries=args.max_queries,
    )
    write_jsonl(predictions, args.output_predictions)

    eval_queries = queries if args.max_queries is None else queries[: args.max_queries]

    metrics = evaluate_qasper_answer_predictions(
        queries=eval_queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote predictions to {args.output_predictions}")
    logger.info(f"Wrote metrics to {args.output_metrics}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Token F1: {metrics['token_f1']:.4f}")


if __name__ == "__main__":
    main()