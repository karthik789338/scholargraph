from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.eval.qasper_answer_metrics import evaluate_qasper_answer_predictions
from src.graph.schemas import Document, Query
from src.utils.io import read_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_queries(path: str | Path) -> List[Query]:
    return [Query(**x) for x in read_jsonl(path)]


def load_documents(path: str | Path) -> List[Document]:
    return [Document(**x) for x in read_jsonl(path)]


def map_documents_by_id(documents: Sequence[Document]) -> Dict[str, Document]:
    return {doc.doc_id: doc for doc in documents}


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def get_gold_answer(query: Query) -> str:
    if query.metadata.is_unanswerable or not query.gold_answer:
        return "unanswerable"
    return normalize_text(query.gold_answer) or "unanswerable"


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
        "Return a short answer phrase or short sentence.\n\n"
        f"Question: {normalize_text(question)}\n\n"
        f"{context}\n\n"
        "Answer:"
    )
    return prompt


def build_examples(
    queries: Sequence[Query],
    documents_by_id: Dict[str, Document],
    max_samples: int | None = None,
) -> List[Dict]:
    examples: List[Dict] = []

    working_queries = list(queries)
    if max_samples is not None:
        working_queries = working_queries[:max_samples]

    for query in working_queries:
        doc = documents_by_id.get(query.source_doc_id)
        if doc is None:
            continue

        prompt = build_prompt(query.text, doc)
        answer = get_gold_answer(query)

        examples.append(
            {
                "query_id": query.query_id,
                "input_text": prompt,
                "target_text": answer,
            }
        )

    return examples


def preprocess_examples(
    batch: Dict[str, List[str]],
    tokenizer,
    max_input_length: int,
    max_target_length: int,
) -> Dict[str, List[List[int]]]:
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]

    # LED benefits from global attention on initial tokens
    global_attention_masks = []
    for input_ids in model_inputs["input_ids"]:
        mask = [0] * len(input_ids)
        for i in range(min(64, len(mask))):
            mask[i] = 1
        global_attention_masks.append(mask)

    model_inputs["global_attention_mask"] = global_attention_masks
    model_inputs["query_id"] = batch["query_id"]

    return model_inputs


def build_eval_metrics_fn(tokenizer, eval_queries: Sequence[Query]):
    query_by_id = {q.query_id: q for q in eval_queries}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Reconstruct pseudo-prediction objects in same order as eval dataset
        predictions_struct = []
        ordered_queries = list(eval_queries)

        for query, pred_text in zip(ordered_queries, pred_texts):
            predictions_struct.append(
                {
                    "query_id": query.query_id,
                    "predicted_answer": normalize_text(pred_text) or "unanswerable",
                    "predicted_evidence_chunks": [],
                }
            )

        metrics = evaluate_qasper_answer_predictions(
            queries=ordered_queries,
            predictions=predictions_struct,
        )

        return {
            "exact_match": metrics["exact_match"],
            "token_f1": metrics["token_f1"],
            "unanswerable_accuracy": metrics["unanswerable_accuracy"],
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train long-context QASPER answer model")
    parser.add_argument("--train-queries", required=True)
    parser.add_argument("--train-documents", required=True)
    parser.add_argument("--validation-queries", required=True)
    parser.add_argument("--validation-documents", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=32)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("Loading train data...")
    train_queries = load_queries(args.train_queries)
    train_documents = load_documents(args.train_documents)
    train_docs_by_id = map_documents_by_id(train_documents)

    logger.info("Loading validation data...")
    val_queries = load_queries(args.validation_queries)
    val_documents = load_documents(args.validation_documents)
    val_docs_by_id = map_documents_by_id(val_documents)

    train_examples = build_examples(
        queries=train_queries,
        documents_by_id=train_docs_by_id,
        max_samples=args.max_train_samples,
    )
    val_examples = build_examples(
        queries=val_queries,
        documents_by_id=val_docs_by_id,
        max_samples=args.max_validation_samples,
    )

    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Optional but usually helpful for long-context fine-tuning
    model.gradient_checkpointing_enable()

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    train_dataset = train_dataset.map(
        lambda batch: preprocess_examples(
            batch=batch,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda batch: preprocess_examples(
            batch=batch,
            tokenizer=tokenizer,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
        ),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    use_fp16 = False
    use_bf16 = False
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        # Trainer will automatically ignore if not supported by the runtime,
        # but we keep the config simple and conservative here.
        use_fp16 = torch.cuda.is_available()

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "fp16": use_fp16,
    }

    # Only add bf16 if supported
    if "bf16" in supported:
        training_kwargs["bf16"] = use_bf16

    # Some versions use evaluation_strategy, some use eval_strategy
    if "evaluation_strategy" in supported:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supported:
        training_kwargs["eval_strategy"] = "steps"

    if "save_strategy" in supported:
        training_kwargs["save_strategy"] = "steps"

    if "eval_steps" in supported:
        training_kwargs["eval_steps"] = args.eval_steps

    if "save_steps" in supported:
        training_kwargs["save_steps"] = args.save_steps

    if "predict_with_generate" in supported:
        training_kwargs["predict_with_generate"] = True

    if "generation_max_length" in supported:
        training_kwargs["generation_max_length"] = args.max_target_length

    if "save_total_limit" in supported:
        training_kwargs["save_total_limit"] = 2

    if "load_best_model_at_end" in supported:
        training_kwargs["load_best_model_at_end"] = True

    if "metric_for_best_model" in supported:
        training_kwargs["metric_for_best_model"] = "token_f1"

    if "greater_is_better" in supported:
        training_kwargs["greater_is_better"] = True

    if "report_to" in supported:
        training_kwargs["report_to"] = "none"

    if "do_train" in supported:
        training_kwargs["do_train"] = True

    if "do_eval" in supported:
        training_kwargs["do_eval"] = True

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    compute_metrics = build_eval_metrics_fn(
        tokenizer=tokenizer,
        eval_queries=val_queries[: args.max_validation_samples] if args.max_validation_samples is not None else val_queries,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Running final evaluation...")
    metrics = trainer.evaluate(max_length=args.max_target_length)
    logger.info(f"Final validation metrics: {metrics}")


if __name__ == "__main__":
    main()