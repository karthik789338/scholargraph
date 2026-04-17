from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.qasper_answer_metrics import evaluate_qasper_answer_predictions
from src.graph.schemas import Chunk, Query
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}

ABSTAIN_STRINGS = {
    "",
    "unanswerable",
    "unknown",
    "cannot answer",
    "can't answer",
    "not answerable",
    "not enough information",
    "insufficient information",
    "insufficient evidence",
    "not mentioned",
    "not provided",
    "not stated",
}

GENERIC_SHORT_ANSWERS = {
    "it",
    "they",
    "this",
    "that",
    "these",
    "those",
    "the paper",
    "the study",
    "the authors",
    "the method",
    "the model",
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "be", "being", "been", "with", "by", "as", "at", "from",
    "this", "that", "these", "those", "it", "they", "their", "its", "we", "our",
}


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_chunks(path: str | Path) -> List[Chunk]:
    return [Chunk(**x) for x in read_jsonl(path)]


def load_evidence_predictions(path: str | Path) -> List[dict]:
    return read_jsonl(path)


def map_chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


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


def normalize_for_match(text: str | None) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = " ".join(text.split())
    return text


def tokenize_simple(text: str | None) -> List[str]:
    return normalize_for_match(text).split()


def content_tokens(text: str | None) -> List[str]:
    return [t for t in tokenize_simple(text) if t not in STOPWORDS]


def token_f1(a: str, b: str) -> float:
    ta = tokenize_simple(a)
    tb = tokenize_simple(b)

    if not ta or not tb:
        return 0.0

    ca = {}
    for tok in ta:
        ca[tok] = ca.get(tok, 0) + 1

    overlap = 0
    for tok in tb:
        if ca.get(tok, 0) > 0:
            overlap += 1
            ca[tok] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def jaccard_content(a: str, b: str) -> float:
    sa = set(content_tokens(a))
    sb = set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_abstain(text: str | None) -> bool:
    norm = normalize_text(text).lower()
    return norm in ABSTAIN_STRINGS


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [p.strip() for p in pieces if p.strip()]


def shorten_answer(text: str, max_words: int = 10) -> str:
    text = normalize_text(text)

    for sep in [".", ";", ":", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).strip()

    return text


def answer_is_generic(text: str) -> bool:
    norm = normalize_text(text).lower()
    if norm in GENERIC_SHORT_ANSWERS:
        return True

    toks = content_tokens(text)
    if not toks:
        return True

    # Too short and generic
    if len(toks) == 1 and toks[0] in {"method", "model", "paper", "study", "approach", "task"}:
        return True

    return False


def build_prompt(question: str, context_records: Sequence[Dict[str, str]]) -> str:
    title = normalize_text(context_records[0].get("title", "")) if context_records else ""

    context_parts = []
    for i, rec in enumerate(context_records):
        sec = normalize_text(rec.get("section_title", ""))
        text = normalize_text(rec.get("text", ""))

        if sec:
            context_parts.append(f"Evidence {i+1} [{sec}]: {text}")
        else:
            context_parts.append(f"Evidence {i+1}: {text}")

    context = "\n\n".join(context_parts)

    prompt = (
        "You are answering a question about a scientific paper.\n"
        "Use only the evidence below.\n"
        "If the answer is not supported by the evidence, return exactly: unanswerable.\n"
        "Otherwise return a short answer phrase copied or closely paraphrased from the evidence.\n"
        "Do not explain.\n\n"
        f"Paper title: {title}\n"
        f"Question: {question}\n\n"
        f"{context}\n\n"
        "Short answer:"
    )
    return prompt


def generate_answers(
    prompts: Sequence[str],
    model_name: str,
    batch_size: int = 8,
    max_input_length: int = 512,
    max_new_tokens: int = 20,
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

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = [normalize_text(x) if normalize_text(x) else "unanswerable" for x in decoded]
        outputs.extend(decoded)

        end = min(start + batch_size, len(prompts))
        if end % 128 == 0 or end == len(prompts):
            logger.info(f"Generated answers for {end}/{len(prompts)} QASPER queries")

    return outputs


def choose_best_sentence(
    context_chunks: Sequence[str],
    question: str,
    generated_answer: str | None = None,
) -> Tuple[str, float]:
    sentences: List[str] = []
    for chunk in context_chunks:
        sentences.extend(split_sentences(chunk))

    if not sentences:
        return "", 0.0

    best_sent = ""
    best_score = -1.0

    for sent in sentences:
        score = 0.9 * token_f1(sent, question) + 0.6 * jaccard_content(sent, question)

        if generated_answer and not is_abstain(generated_answer):
            score += 1.3 * token_f1(sent, generated_answer)
            score += 0.5 * jaccard_content(sent, generated_answer)

            if normalize_for_match(generated_answer) in normalize_for_match(sent):
                score += 0.25

        if score > best_score:
            best_score = score
            best_sent = sent

    return best_sent, best_score


def best_phrase_from_sentence(
    sentence: str,
    question: str,
    generated_answer: str | None = None,
    max_ngram_words: int = 10,
) -> Tuple[str, float]:
    tokens = sentence.split()
    if not tokens:
        return "", 0.0

    best_phrase = ""
    best_score = -1.0

    upper = min(max_ngram_words, len(tokens))
    for n in range(1, upper + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n]).strip()
            if not phrase:
                continue

            score = 0.5 * token_f1(phrase, question) + 0.3 * jaccard_content(phrase, question)

            if generated_answer and not is_abstain(generated_answer):
                score += 1.6 * token_f1(phrase, generated_answer)
                score += 0.5 * jaccard_content(phrase, generated_answer)

                if normalize_for_match(phrase) == normalize_for_match(generated_answer):
                    score += 0.35
                elif normalize_for_match(phrase) in normalize_for_match(generated_answer):
                    score += 0.20

            # slight preference for shorter spans
            score -= 0.01 * n

            if score > best_score:
                best_score = score
                best_phrase = phrase

    return normalize_text(best_phrase), best_score


def refine_answer_with_answerability_gate(
    generated_answer: str,
    context_chunks: Sequence[str],
    question: str,
    sentence_support_threshold: float = 0.40,
    phrase_support_threshold: float = 0.30,
) -> Tuple[str, Dict[str, float]]:
    """
    Extractive-first refinement with abstention.
    """
    generated_answer = normalize_text(generated_answer)

    best_sentence, sentence_score = choose_best_sentence(
        context_chunks=context_chunks,
        question=question,
        generated_answer=generated_answer,
    )

    if not best_sentence:
        return "unanswerable", {
            "best_sentence_score": 0.0,
            "best_phrase_score": 0.0,
        }

    best_phrase, phrase_score = best_phrase_from_sentence(
        sentence=best_sentence,
        question=question,
        generated_answer=generated_answer,
        max_ngram_words=10,
    )

    diagnostics = {
        "best_sentence_score": sentence_score,
        "best_phrase_score": phrase_score,
    }

    # Hard abstain if nothing looks supported
    if sentence_score < sentence_support_threshold:
        return "unanswerable", diagnostics

    if is_abstain(generated_answer):
        # only allow extractive fallback if phrase support is decent
        if phrase_score >= phrase_support_threshold and not answer_is_generic(best_phrase):
            return shorten_answer(best_phrase, max_words=10), diagnostics
        return "unanswerable", diagnostics

    # If generation is long / noisy, prefer extractive phrase if supported
    if phrase_score >= phrase_support_threshold and not answer_is_generic(best_phrase):
        return shorten_answer(best_phrase, max_words=10), diagnostics

    # fallback to shortened generation only if it appears supported by context
    if sentence_score >= sentence_support_threshold + 0.08:
        short_gen = shorten_answer(generated_answer, max_words=10)
        if not answer_is_generic(short_gen):
            return short_gen, diagnostics

    return "unanswerable", diagnostics


def build_predictions(
    queries: Sequence[Query],
    evidence_predictions: Sequence[dict],
    chunks_by_id: Dict[str, Chunk],
    model_name: str = "google/flan-t5-small",
    batch_size: int = 8,
    max_input_length: int = 512,
    max_new_tokens: int = 20,
) -> List[Dict]:
    pred_by_query_id = {pred["query_id"]: pred for pred in evidence_predictions}
    query_by_id = {q.query_id: q for q in queries}

    prompts: List[str] = []
    meta: List[Tuple[str, List[str], List[str]]] = []

    for query in queries:
        evidence_pred = pred_by_query_id.get(query.query_id, {})
        evidence_chunk_ids = evidence_pred.get("predicted_evidence_chunks", [])

        context_records = []
        raw_context_texts = []

        for cid in evidence_chunk_ids:
            chunk = chunks_by_id.get(cid)
            if chunk is None or not chunk.text.strip():
                continue

            context_records.append(
                {
                    "title": normalize_text(chunk.metadata.title if chunk.metadata else ""),
                    "section_title": normalize_text(chunk.section_title),
                    "text": normalize_text(chunk.text),
                }
            )
            raw_context_texts.append(chunk.text)

        if not context_records:
            prompt = (
                "You are answering a question about a scientific paper.\n"
                "If the evidence does not support an answer, return exactly: unanswerable.\n\n"
                f"Question: {query.text}\n\n"
                "Short answer:"
            )
        else:
            prompt = build_prompt(query.text, context_records)

        prompts.append(prompt)
        meta.append((query.query_id, evidence_chunk_ids, raw_context_texts))

    logger.info(f"Generating answers for {len(prompts)} QASPER queries with model={model_name}")
    generated_answers = generate_answers(
        prompts=prompts,
        model_name=model_name,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )

    predictions = []
    for (query_id, evidence_chunk_ids, raw_context_texts), generated_answer in zip(meta, generated_answers):
        query_text = query_by_id[query_id].text

        refined_answer, diagnostics = refine_answer_with_answerability_gate(
            generated_answer=generated_answer,
            context_chunks=raw_context_texts,
            question=query_text,
            sentence_support_threshold=0.40,
            phrase_support_threshold=0.30,
        )

        predictions.append(
            {
                "query_id": query_id,
                "predicted_answer": refined_answer,
                "raw_generated_answer": generated_answer,
                "predicted_evidence_chunks": evidence_chunk_ids,
                "diagnostics": diagnostics,
            }
        )

    logger.info(f"Built {len(predictions)} QASPER answer predictions")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run QASPER answerability + extractive baseline")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--evidence-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)

    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=20)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    chunks = load_chunks(args.chunks)
    evidence_predictions = load_evidence_predictions(args.evidence_predictions)
    chunks_by_id = map_chunks_by_id(chunks)

    predictions = build_predictions(
        queries=queries,
        evidence_predictions=evidence_predictions,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(predictions, args.output_predictions)

    metrics = evaluate_qasper_answer_predictions(
        queries=queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote predictions to {args.output_predictions}")
    logger.info(f"Wrote metrics to {args.output_metrics}")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"Token F1: {metrics['token_f1']:.4f}")


if __name__ == "__main__":
    main()