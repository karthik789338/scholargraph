from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.graph.schemas import Document, DocumentMetadata, GoldEvidence, Query, QueryMetadata, Section
from src.utils.hashing import make_doc_id, make_query_id, make_section_id
from src.utils.io import read_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_qasper(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = read_json(path)
    logger.info(f"Loaded QASPER raw file from {path}")
    return data


def _normalize_abstract(abstract_value: Any) -> str | None:
    if abstract_value is None:
        return None

    if isinstance(abstract_value, str):
        text = abstract_value.strip()
        return text if text else None

    if isinstance(abstract_value, list):
        parts = [str(x).strip() for x in abstract_value if str(x).strip()]
        text = " ".join(parts).strip()
        return text if text else None

    text = str(abstract_value).strip()
    return text if text else None


def _extract_sections(full_text: Any, doc_id: str) -> Tuple[List[Section], str | None]:
    sections: List[Section] = []
    full_text_parts: List[str] = []

    # Old format: list of section dicts
    if isinstance(full_text, list):
        for i, sec in enumerate(full_text):
            if not isinstance(sec, dict):
                continue

            section_name = (
                sec.get("section_name")
                or sec.get("section_title")
                or f"section_{i}"
            )

            paragraphs = sec.get("paragraphs", [])
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]

            section_text = "\n".join(
                str(p).strip() for p in paragraphs if str(p).strip()
            ).strip()

            if not section_text:
                continue

            section = Section(
                section_id=make_section_id(doc_id, i),
                section_title=section_name,
                section_text=section_text,
            )
            sections.append(section)
            full_text_parts.append(f"{section_name}\n{section_text}")

    # Parquet/HF-converted format: dict of lists
    elif isinstance(full_text, dict):
        section_names = full_text.get("section_name", []) or full_text.get("section_title", [])
        paragraphs_list = full_text.get("paragraphs", [])

        n = max(len(section_names), len(paragraphs_list))

        for i in range(n):
            section_name = (
                str(section_names[i]).strip()
                if i < len(section_names) and str(section_names[i]).strip()
                else f"section_{i}"
            )

            paragraphs = paragraphs_list[i] if i < len(paragraphs_list) else []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]

            section_text = "\n".join(
                str(p).strip() for p in paragraphs if str(p).strip()
            ).strip()

            if not section_text:
                continue

            section = Section(
                section_id=make_section_id(doc_id, i),
                section_title=section_name,
                section_text=section_text,
            )
            sections.append(section)
            full_text_parts.append(f"{section_name}\n{section_text}")

    full_text_str = "\n\n".join(full_text_parts).strip()
    return sections, (full_text_str if full_text_str else None)


def normalize_qasper_papers(raw: Dict[str, Any]) -> List[Document]:
    documents: List[Document] = []

    for source_id, item in raw.items():
        if not isinstance(item, dict):
            continue

        doc_id = make_doc_id("qasper", str(source_id))
        title = str(item.get("title", "")).strip()
        abstract = _normalize_abstract(item.get("abstract"))
        sections, full_text = _extract_sections(item.get("full_text", []), doc_id)

        doc = Document(
            doc_id=doc_id,
            dataset="qasper",
            title=title,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            metadata=DocumentMetadata(
                authors=[str(a).strip() for a in item.get("authors", []) if str(a).strip()],
                year=item.get("year", None),
                venue=item.get("venue", None),
                domain="scientific",
            ),
        )
        documents.append(doc)

    logger.info(f"Normalized {len(documents)} QASPER documents")
    return documents


def _extract_answer_and_flags(answer_obj: Dict[str, Any]) -> Tuple[str | None, bool]:
    if not isinstance(answer_obj, dict):
        return None, False

    if answer_obj.get("unanswerable", False) is True:
        return None, True

    free_form = answer_obj.get("free_form_answer")
    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip(), False
    if isinstance(free_form, list):
        parts = [str(x).strip() for x in free_form if str(x).strip()]
        if parts:
            return " ".join(parts), False

    spans = answer_obj.get("extractive_spans")
    if isinstance(spans, list):
        parts = [str(x).strip() for x in spans if str(x).strip()]
        if parts:
            return " ".join(parts), False

    yes_no = answer_obj.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes" if yes_no else "no", False
    if isinstance(yes_no, str) and yes_no.strip():
        return yes_no.strip().lower(), False

    return None, False


def _iter_qasper_qas(qas: Any) -> List[Dict[str, Any]]:
    """
    Support both:
    1. old format: list[dict]
    2. parquet format: dict of parallel lists
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

def normalize_qasper_queries(raw: Dict[str, Any]) -> List[Query]:
    queries: List[Query] = []

    for source_id, item in raw.items():
        if not isinstance(item, dict):
            continue

        doc_id = make_doc_id("qasper", str(source_id))
        qas = _iter_qasper_qas(item.get("qas", []))

        for q_idx, qa in enumerate(qas):
            question = str(qa.get("question", "")).strip()
            if not question:
                continue

            answers = _iter_qasper_answers(qa.get("answers", []))

            gold_answer = None
            is_unanswerable = False
            gold_evidence: List[GoldEvidence] = []

            if answers:
                first_answer = answers[0]
                gold_answer, is_unanswerable = _extract_answer_and_flags(first_answer)

            query = Query(
                query_id=make_query_id("qasper", f"{source_id}_{q_idx}"),
                task_type="qa",
                dataset="qasper",
                doc_scope="closed",
                text=question,
                source_doc_id=doc_id,
                gold_answer=gold_answer,
                gold_label=None,
                gold_evidence=gold_evidence,
                metadata=QueryMetadata(
                    question_type=str(qa.get("question_type")).strip() if qa.get("question_type") is not None else None,
                    is_unanswerable=is_unanswerable,
                ),
            )
            queries.append(query)

    logger.info(f"Normalized {len(queries)} QASPER queries")
    return queries