from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.eval.scifact_metrics import evaluate_scifact_predictions
from src.graph.schemas import Chunk, GraphInput, Query
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForSequenceClassification]] = {}


def load_queries(path: str | Path) -> List[Query]:
    records = read_jsonl(path)
    return [Query(**record) for record in records]


def load_graph_inputs(path: str | Path) -> List[GraphInput]:
    records = read_jsonl(path)
    return [GraphInput(**record) for record in records]


def load_chunks(path: str | Path) -> List[Chunk]:
    records = read_jsonl(path)
    return [Chunk(**record) for record in records]


def map_chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_nli_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(get_device())
    model.eval()

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def normalize_nli_label(label: str) -> str:
    x = label.lower().strip()
    if "entail" in x:
        return "entailment"
    if "contrad" in x:
        return "contradiction"
    if "neutral" in x:
        return "neutral"
    return x


def get_label_index_map(model) -> Dict[str, int]:
    id2label = model.config.id2label
    normalized = {}
    for idx, label in id2label.items():
        normalized[normalize_nli_label(str(label))] = int(idx)

    required = {"entailment", "contradiction", "neutral"}
    missing = required - set(normalized.keys())
    if missing:
        raise ValueError(f"NLI model labels missing expected classes: {missing}. Found: {normalized}")

    return normalized

def select_top_chunks_for_nli(
    graph_input: GraphInput,
    top_nli_chunks: int = 3,
) -> List[str]:
    """
    Select only the top retrieval-ranked chunks for NLI scoring.
    Uses retrieval rank first, then retrieval score as tie-breaker.
    """
    retrieval_scores = graph_input.metadata.get("retrieval_scores", {})

    enriched = []
    for chunk_id in graph_input.candidate_chunks:
        info = retrieval_scores.get(chunk_id, {})
        rank = int(info.get("rank", 10**9))
        score = float(info.get("score", 0.0))
        enriched.append((rank, -score, chunk_id))

    enriched.sort()
    selected = [chunk_id for _, _, chunk_id in enriched[:top_nli_chunks]]
    return selected

def score_claim_evidence_pairs(
    claim_evidence_pairs: Sequence[Tuple[str, str]],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 256,
) -> List[Dict[str, float]]:
    tokenizer, model = load_nli_model(model_name)
    label_index_map = get_label_index_map(model)
    device = get_device()

    all_scores: List[Dict[str, float]] = []
    total = len(claim_evidence_pairs)

    logger.info(
        f"Starting NLI scoring on device={device} with batch_size={batch_size}, "
        f"max_length={max_length}, total_pairs={total}"
    )

    for start in range(0, total, batch_size):
        batch = claim_evidence_pairs[start : start + batch_size]

        premises = [evidence for claim, evidence in batch]
        hypotheses = [claim for claim, evidence in batch]

        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu()

        for row in probs:
            all_scores.append(
                {
                    "supports": float(row[label_index_map["entailment"]].item()),
                    "refutes": float(row[label_index_map["contradiction"]].item()),
                    "neutral": float(row[label_index_map["neutral"]].item()),
                }
            )

        end = min(start + batch_size, total)
        if end % 128 == 0 or end == total:
            logger.info(f"Scored {end}/{total} claim-evidence pairs")

    return all_scores


def choose_verdict_and_evidence(
    scored_chunks: Sequence[Dict],
    label_threshold: float = 0.55,
    margin_threshold: float = 0.05,
    max_evidence_chunks: int = 1,
) -> Tuple[str, List[str], Dict[str, float]]:
    if not scored_chunks:
        return "insufficient", [], {
            "best_support_score": 0.0,
            "best_refute_score": 0.0,
            "best_neutral_score": 1.0,
        }

    best_support = max(scored_chunks, key=lambda x: x["supports"])
    best_refute = max(scored_chunks, key=lambda x: x["refutes"])
    best_neutral = max(scored_chunks, key=lambda x: x["neutral"])

    support_score = float(best_support["supports"])
    refute_score = float(best_refute["refutes"])
    neutral_score = float(best_neutral["neutral"])

    win_score = max(support_score, refute_score)
    margin = abs(support_score - refute_score)

    if win_score < label_threshold:
        return "insufficient", [], {
            "best_support_score": support_score,
            "best_refute_score": refute_score,
            "best_neutral_score": neutral_score,
        }

    if margin < margin_threshold and win_score < (label_threshold + 0.10):
        return "insufficient", [], {
            "best_support_score": support_score,
            "best_refute_score": refute_score,
            "best_neutral_score": neutral_score,
        }

    if support_score >= refute_score:
        predicted_label = "supports"
        ranked = sorted(scored_chunks, key=lambda x: x["supports"], reverse=True)
    else:
        predicted_label = "refutes"
        ranked = sorted(scored_chunks, key=lambda x: x["refutes"], reverse=True)

    predicted_evidence_chunks = [item["chunk_id"] for item in ranked[:max_evidence_chunks]]

    return predicted_label, predicted_evidence_chunks, {
        "best_support_score": support_score,
        "best_refute_score": refute_score,
        "best_neutral_score": neutral_score,
    }


def build_predictions(
    queries: Sequence[Query],
    graph_inputs: Sequence[GraphInput],
    chunks_by_id: Dict[str, Chunk],
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    batch_size: int = 32,
    max_length: int = 512,
    label_threshold: float = 0.55,
    margin_threshold: float = 0.05,
    max_evidence_chunks: int = 1,
    top_nli_chunks: int = 3,
) -> List[Dict]:
    query_by_id = {query.query_id: query for query in queries}

    pair_texts: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str]] = []

    total_candidate_chunks = 0
    total_scored_chunks = 0

    for graph_input in graph_inputs:
        query = query_by_id[graph_input.query_id]

        total_candidate_chunks += len(graph_input.candidate_chunks)
        selected_chunk_ids = select_top_chunks_for_nli(
            graph_input=graph_input,
            top_nli_chunks=top_nli_chunks,
        )
        total_scored_chunks += len(selected_chunk_ids)

        for chunk_id in selected_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            pair_texts.append((query.text, chunk.text))
            pair_meta.append((query.query_id, chunk_id))

    logger.info(
        f"Selected {total_scored_chunks} chunks for NLI scoring "
        f"(from {total_candidate_chunks} retrieved chunks total)"
    )

    logger.info(f"Scoring {len(pair_texts)} claim-evidence pairs with NLI baseline")
    all_scores = score_claim_evidence_pairs(
        claim_evidence_pairs=pair_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    scored_by_query: Dict[str, List[Dict]] = {}

    for (query_id, chunk_id), scores in zip(pair_meta, all_scores):
        scored_by_query.setdefault(query_id, []).append(
            {
                "chunk_id": chunk_id,
                "supports": scores["supports"],
                "refutes": scores["refutes"],
                "neutral": scores["neutral"],
            }
        )

    predictions: List[Dict] = []

    for graph_input in graph_inputs:
        scored_chunks = scored_by_query.get(graph_input.query_id, [])

        predicted_label, predicted_evidence_chunks, score_summary = choose_verdict_and_evidence(
            scored_chunks=scored_chunks,
            label_threshold=label_threshold,
            margin_threshold=margin_threshold,
            max_evidence_chunks=max_evidence_chunks,
        )

        # keep top scored chunk details for later error analysis
        top_chunks_sorted = sorted(
            scored_chunks,
            key=lambda x: max(x["supports"], x["refutes"], x["neutral"]),
            reverse=True,
        )[:5]

        predictions.append(
            {
                "query_id": graph_input.query_id,
                "predicted_label": predicted_label,
                "predicted_evidence_chunks": predicted_evidence_chunks,
                "score_summary": score_summary,
                "top_chunk_scores": top_chunks_sorted,
                "candidate_chunk_count": len(graph_input.candidate_chunks),
                "scored_chunk_count": min(top_nli_chunks, len(graph_input.candidate_chunks)),
            }
        )

    logger.info(f"Built {len(predictions)} SciFact baseline predictions")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciFact retrieval+NLI baseline.")
    parser.add_argument("--queries", required=True, help="Path to Query JSONL")
    parser.add_argument("--graph-inputs", required=True, help="Path to GraphInput JSONL")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")

    parser.add_argument("--output-predictions", required=True, help="Path to save predictions JSONL")
    parser.add_argument("--output-metrics", required=True, help="Path to save metrics JSON")

    parser.add_argument(
        "--model-name",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HF NLI model name",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--label-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--max-evidence-chunks", type=int, default=1)
    parser.add_argument(
        "--top-nli-chunks",
        type=int,
        default=3,
        help="How many top retrieved chunks to send to NLI scoring per query",
    )

    args = parser.parse_args()

    queries = load_queries(args.queries)
    graph_inputs = load_graph_inputs(args.graph_inputs)
    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)

    predictions = build_predictions(
        queries=queries,
        graph_inputs=graph_inputs,
        chunks_by_id=chunks_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label_threshold=args.label_threshold,
        margin_threshold=args.margin_threshold,
        max_evidence_chunks=args.max_evidence_chunks,
        top_nli_chunks=args.top_nli_chunks,
    )

    write_jsonl(predictions, args.output_predictions)

    metrics = evaluate_scifact_predictions(
        queries=queries,
        predictions=predictions,
    )
    write_json(metrics, args.output_metrics)

    logger.info(f"Wrote predictions to {args.output_predictions}")
    logger.info(f"Wrote metrics to {args.output_metrics}")
    logger.info(f"Label macro F1: {metrics['label_metrics']['macro_f1']:.4f}")
    logger.info(f"Evidence micro F1: {metrics['evidence_metrics']['micro_f1']:.4f}")


if __name__ == "__main__":
    main()