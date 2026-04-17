from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.graph.schemas import Chunk, GoldEvidence, Query
from src.utils.io import read_json, read_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

_QASPER_QUERY_RE = re.compile(r"^qasper_query_(.+)_(\d+)$")


def normalize_match_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text: str | None) -> set[str]:
    norm = normalize_match_text(text)
    if not norm:
        return set()
    return set(norm.split())


def overlap_score(a: str | None, b: str | None) -> float:
    """
    Simple token-overlap score.
    """
    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def map_chunks_by_doc(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
    grouped: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    for doc_id in grouped:
        grouped[doc_id] = sorted(grouped[doc_id], key=lambda x: x.chunk_index)

    return grouped


def map_sentence_to_chunk_ids(chunks: Sequence[Chunk]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for chunk in chunks:
        for sentence_id in chunk.sentence_ids:
            mapping.setdefault(sentence_id, []).append(chunk.chunk_id)
    return mapping


def map_chunk_id_to_chunk(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def choose_best_chunk_for_sentences(
    sentence_ids: Sequence[str],
    sentence_to_chunk_ids: Dict[str, List[str]],
) -> Optional[str]:
    """
    Choose the chunk that covers the most gold evidence sentences.
    Useful for SciFact where one rationale may span multiple sentences.
    """
    votes: Dict[str, int] = {}

    for sent_id in sentence_ids:
        for chunk_id in sentence_to_chunk_ids.get(sent_id, []):
            votes[chunk_id] = votes.get(chunk_id, 0) + 1

    if not votes:
        return None

    # Highest coverage, then stable lexical order
    best_chunk_id = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best_chunk_id


def choose_best_chunk_for_text(
    evidence_text: str,
    doc_chunks: Sequence[Chunk],
) -> Optional[str]:
    """
    Best-effort text-to-chunk matcher for QASPER evidence paragraphs.
    """
    if not evidence_text or not evidence_text.strip():
        return None

    norm_evidence = normalize_match_text(evidence_text)
    best_chunk_id: Optional[str] = None
    best_score = 0.0

    for chunk in doc_chunks:
        norm_chunk = normalize_match_text(chunk.text)

        if not norm_chunk:
            continue

        # Strong match if one contains the other
        if norm_evidence in norm_chunk or norm_chunk in norm_evidence:
            score = 1.0
        else:
            score = overlap_score(norm_evidence, norm_chunk)

        if score > best_score:
            best_score = score
            best_chunk_id = chunk.chunk_id

    # Prevent very weak accidental matches
    if best_score < 0.10:
        return None

    return best_chunk_id


def rebuild_query_with_gold_evidence(query: Query, gold_evidence: List[GoldEvidence]) -> Query:
    return Query(
        query_id=query.query_id,
        task_type=query.task_type,
        dataset=query.dataset,
        doc_scope=query.doc_scope,
        text=query.text,
        source_doc_id=query.source_doc_id,
        gold_answer=query.gold_answer,
        gold_label=query.gold_label,
        gold_evidence=gold_evidence,
        metadata=query.metadata,
    )


def attach_gold_evidence_scifact(
    queries: Sequence[Query],
    chunks: Sequence[Chunk],
) -> List[Query]:
    """
    Map SciFact gold sentence IDs to chunk IDs.
    """
    sentence_to_chunk_ids = map_sentence_to_chunk_ids(chunks)
    updated_queries: List[Query] = []

    for query in queries:
        if query.dataset != "scifact":
            updated_queries.append(query)
            continue

        new_gold_evidence: List[GoldEvidence] = []
        for ev in query.gold_evidence:
            best_chunk_id = choose_best_chunk_for_sentences(ev.sentence_ids, sentence_to_chunk_ids)

            new_gold_evidence.append(
                GoldEvidence(
                    doc_id=ev.doc_id,
                    chunk_id=best_chunk_id,
                    sentence_ids=ev.sentence_ids,
                )
            )

        updated_queries.append(rebuild_query_with_gold_evidence(query, new_gold_evidence))

    logger.info(f"Attached chunk-level gold evidence for {len(updated_queries)} SciFact queries")
    return updated_queries


def _extract_qasper_query_parts(query_id: str) -> Optional[Tuple[str, int]]:
    """
    qasper_query_<source_id>_<q_idx>
    """
    match = _QASPER_QUERY_RE.match(query_id)
    if not match:
        return None

    source_id = match.group(1)
    q_idx = int(match.group(2))
    return source_id, q_idx


def _flatten_qasper_paragraphs(raw_doc_item: Dict[str, Any]) -> List[str]:
    """
    Flatten QASPER paragraphs in source order.
    Supports both:
    1. old format: full_text = list[dict(section_name, paragraphs)]
    2. parquet format: full_text = {"section_name": [...], "paragraphs": [[...], ...]}
    """
    paragraphs: List[str] = []
    full_text = raw_doc_item.get("full_text", [])

    # old format
    if isinstance(full_text, list):
        for section in full_text:
            if not isinstance(section, dict):
                continue
            for para in section.get("paragraphs", []):
                if isinstance(para, str) and para.strip():
                    paragraphs.append(para.strip())

    # parquet format
    elif isinstance(full_text, dict):
        paragraphs_list = full_text.get("paragraphs", [])
        for section_paragraphs in paragraphs_list:
            if isinstance(section_paragraphs, str):
                section_paragraphs = [section_paragraphs]
            if isinstance(section_paragraphs, list):
                for para in section_paragraphs:
                    if isinstance(para, str) and para.strip():
                        paragraphs.append(para.strip())

    return paragraphs


def _coerce_to_flat_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(_coerce_to_flat_list(item))
        return flattened

    return [value]


def _extract_qasper_evidence_texts_from_answer(
    answer_obj: Dict[str, Any],
    flattened_paragraphs: Sequence[str],
) -> List[str]:
    """
    Robustly recover evidence texts from a QASPER answer object.

    QASPER releases can vary a bit in how evidence is stored, so we support:
    - paragraph indices
    - raw paragraph text
    - nested lists
    - dict wrappers with text-like fields
    """
    evidence_raw = answer_obj.get("evidence", [])
    raw_items = _coerce_to_flat_list(evidence_raw)

    evidence_texts: List[str] = []

    for item in raw_items:
        # int paragraph index
        if isinstance(item, int):
            if 0 <= item < len(flattened_paragraphs):
                evidence_texts.append(flattened_paragraphs[item])
            continue

        # numeric string paragraph index
        if isinstance(item, str):
            stripped = item.strip()
            if not stripped:
                continue

            if stripped.isdigit():
                idx = int(stripped)
                if 0 <= idx < len(flattened_paragraphs):
                    evidence_texts.append(flattened_paragraphs[idx])
            else:
                evidence_texts.append(stripped)
            continue

        # dict with possible text keys
        if isinstance(item, dict):
            for key in ["text", "paragraph_text", "paragraph", "evidence_text"]:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    evidence_texts.append(value.strip())
                    break

    # deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for text in evidence_texts:
        norm = normalize_match_text(text)
        if norm and norm not in seen:
            deduped.append(text)
            seen.add(norm)

    return deduped

def _iter_qasper_qas(qas: Any) -> List[Dict[str, Any]]:
    """
    Support both:
    1. old format: list[dict]
    2. parquet/HF format: dict of parallel lists
    """
    if isinstance(qas, list):
        return [x for x in qas if isinstance(x, dict)]

    if isinstance(qas, dict):
        question_ids = qas.get("question_id", [])
        questions = qas.get("question", [])
        answers = qas.get("answers", [])

        n = max(len(question_ids), len(questions), len(answers))
        out = []

        for i in range(n):
            out.append(
                {
                    "question_id": question_ids[i] if i < len(question_ids) else i,
                    "question": questions[i] if i < len(questions) else "",
                    "answers": answers[i] if i < len(answers) else [],
                }
            )
        return out

    return []

def _iter_qasper_answers(answers: Any) -> List[Dict[str, Any]]:
    """
    Support both:
    1. old format: list[dict]
    2. parquet format: {"answer": [...], "annotation_id": [...], ...}
    """
    if isinstance(answers, list):
        return [x for x in answers if isinstance(x, dict)]

    if isinstance(answers, dict):
        answer_list = answers.get("answer", [])
        if isinstance(answer_list, list):
            return [x for x in answer_list if isinstance(x, dict)]

    return []

def build_qasper_gold_evidence_text_index(raw_qasper: Dict[str, Any]) -> Dict[Tuple[str, int], List[str]]:
    """
    Build mapping:
        (source_id, question_index) -> list[evidence_paragraph_text]
    Supports both old QASPER JSON and parquet/HF-converted JSON.
    """
    result: Dict[Tuple[str, int], List[str]] = {}

    for source_id, item in raw_qasper.items():
        flattened_paragraphs = _flatten_qasper_paragraphs(item)
        qas = _iter_qasper_qas(item.get("qas", []))

        for q_idx, qa in enumerate(qas):
            answers = _iter_qasper_answers(qa.get("answers", []))
            evidence_texts: List[str] = []

            if answers:
                first_answer = answers[0]

                if isinstance(first_answer, dict):
                    evidence_texts = _extract_qasper_evidence_texts_from_answer(
                        answer_obj=first_answer,
                        flattened_paragraphs=flattened_paragraphs,
                    )

            result[(str(source_id), q_idx)] = evidence_texts

    return result


def attach_gold_evidence_qasper(
    raw_qasper: Dict[str, Any],
    queries: Sequence[Query],
    chunks: Sequence[Chunk],
) -> List[Query]:
    """
    Map QASPER gold evidence paragraphs to chunk IDs using best-effort text matching.
    """
    by_doc = map_chunks_by_doc(chunks)
    evidence_index = build_qasper_gold_evidence_text_index(raw_qasper)

    updated_queries: List[Query] = []

    for query in queries:
        if query.dataset != "qasper":
            updated_queries.append(query)
            continue

        parsed = _extract_qasper_query_parts(query.query_id)
        if parsed is None:
            logger.warning(f"Could not parse QASPER query_id: {query.query_id}")
            updated_queries.append(query)
            continue

        source_id, q_idx = parsed
        evidence_texts = evidence_index.get((source_id, q_idx), [])

        doc_chunks = by_doc.get(query.source_doc_id or "", [])
        new_gold_evidence: List[GoldEvidence] = []

        for ev_text in evidence_texts:
            chunk_id = choose_best_chunk_for_text(ev_text, doc_chunks)
            new_gold_evidence.append(
                GoldEvidence(
                    doc_id=query.source_doc_id or "",
                    chunk_id=chunk_id,
                    sentence_ids=[],
                )
            )

        updated_queries.append(rebuild_query_with_gold_evidence(query, new_gold_evidence))

    logger.info(f"Attached chunk-level gold evidence for {len(updated_queries)} QASPER queries")
    return updated_queries


def validate_queries_have_valid_chunk_ids(
    queries: Sequence[Query],
    chunks: Sequence[Chunk],
) -> Dict[str, int]:
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    total_evidence = 0
    missing_chunk_refs = 0

    for query in queries:
        for ev in query.gold_evidence:
            total_evidence += 1
            if ev.chunk_id is not None and ev.chunk_id not in chunk_ids:
                missing_chunk_refs += 1

    stats = {
        "total_queries": len(queries),
        "total_gold_evidence_items": total_evidence,
        "missing_chunk_refs": missing_chunk_refs,
    }
    return stats


def write_queries_jsonl(queries: Iterable[Query], output_path: str | Path) -> None:
    write_jsonl(queries, output_path)
    logger.info(f"Wrote queries to {output_path}")


def _load_queries_file(path: str | Path) -> List[Query]:
    records = read_jsonl(path)
    return [Query(**record) for record in records]


def _load_chunks_file(path: str | Path) -> List[Chunk]:
    records = read_jsonl(path)
    return [Chunk(**record) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach chunk-level gold evidence to queries.")
    parser.add_argument("--dataset", choices=["qasper", "scifact", "scifact_open"], required=True)
    parser.add_argument("--queries", required=True, help="Path to normalized queries JSONL")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")
    parser.add_argument("--output", required=True, help="Path to write enriched queries JSONL")

    # Only needed for QASPER
    parser.add_argument("--raw-qasper", default=None, help="Path to raw QASPER JSON")

    args = parser.parse_args()

    queries = _load_queries_file(args.queries)
    chunks = _load_chunks_file(args.chunks)

    if args.dataset in {"scifact", "scifact_open"}:
        updated_queries = attach_gold_evidence_scifact(queries, chunks)
    elif args.dataset == "qasper":
        if not args.raw_qasper:
            raise ValueError("--raw-qasper is required for dataset=qasper")
        raw_qasper = load_raw_qasper_json(args.raw_qasper)
        updated_queries = attach_gold_evidence_qasper(raw_qasper, queries, chunks)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    stats = validate_queries_have_valid_chunk_ids(updated_queries, chunks)
    logger.info(f"Validation stats: {stats}")

    write_queries_jsonl(updated_queries, args.output)


if __name__ == "__main__":
    main()