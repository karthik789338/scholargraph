from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from src.graph.schemas import Document, DocumentMetadata, Query, Section


_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str | None) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    if text is None:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WHITESPACE_RE.sub(" ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = _NEWLINE_RE.sub("\n\n", text).strip()
    return text


def normalize_inline_text(text: str | None) -> str:
    """
    Normalize text to a single line.
    Useful for titles, claims, short answers, and metadata fields.
    """
    if text is None:
        return ""
    text = normalize_whitespace(text)
    return " ".join(text.split()).strip()


def normalize_section_title(title: str | None, fallback: str = "untitled_section") -> str:
    title = normalize_inline_text(title)
    return title if title else fallback


def normalize_authors(authors: Sequence[str] | None) -> List[str]:
    if not authors:
        return []
    cleaned = [normalize_inline_text(a) for a in authors]
    return [a for a in cleaned if a]


def materialize_full_text(sections: Sequence[Section]) -> str | None:
    """
    Build full_text from sections if possible.
    """
    if not sections:
        return None

    parts: List[str] = []
    for sec in sections:
        title = normalize_section_title(sec.section_title, fallback="")
        body = normalize_whitespace(sec.section_text)
        if not body:
            continue

        if title:
            parts.append(f"{title}\n{body}")
        else:
            parts.append(body)

    full_text = "\n\n".join(parts).strip()
    return full_text if full_text else None


def normalize_sections(sections: Sequence[Section] | None) -> List[Section]:
    normalized: List[Section] = []

    if not sections:
        return normalized

    for idx, sec in enumerate(sections):
        section_text = normalize_whitespace(sec.section_text)
        if not section_text:
            continue

        normalized.append(
            Section(
                section_id=sec.section_id,
                section_title=normalize_section_title(sec.section_title, fallback=f"section_{idx}"),
                section_text=section_text,
            )
        )

    return normalized


def ensure_document_consistency(doc: Document) -> Document:
    """
    Return a cleaned, consistent Document object.
    """
    sections = normalize_sections(doc.sections)

    title = normalize_inline_text(doc.title)
    abstract = normalize_whitespace(doc.abstract) if doc.abstract else None
    full_text = normalize_whitespace(doc.full_text) if doc.full_text else None

    if not full_text:
        full_text = materialize_full_text(sections)

    metadata = DocumentMetadata(
        year=doc.metadata.year,
        venue=normalize_inline_text(doc.metadata.venue) or None,
        authors=normalize_authors(doc.metadata.authors),
        source_url=normalize_inline_text(doc.metadata.source_url) or None,
        domain=normalize_inline_text(doc.metadata.domain) or None,
    )

    return Document(
        doc_id=doc.doc_id,
        dataset=doc.dataset,
        title=title,
        abstract=abstract,
        full_text=full_text,
        sections=sections,
        metadata=metadata,
    )


def ensure_documents_consistency(documents: Iterable[Document]) -> List[Document]:
    return [ensure_document_consistency(doc) for doc in documents]


def ensure_query_consistency(query: Query) -> Query:
    """
    Clean query text and answer text without changing labels/evidence.
    """
    return Query(
        query_id=query.query_id,
        task_type=query.task_type,
        dataset=query.dataset,
        doc_scope=query.doc_scope,
        text=normalize_inline_text(query.text),
        source_doc_id=query.source_doc_id,
        gold_answer=normalize_inline_text(query.gold_answer) or None if query.gold_answer is not None else None,
        gold_label=query.gold_label,
        gold_evidence=query.gold_evidence,
        metadata=query.metadata,
    )


def ensure_queries_consistency(queries: Iterable[Query]) -> List[Query]:
    return [ensure_query_consistency(q) for q in queries]


def validate_unique_ids(items: Sequence[object], attr_name: str) -> None:
    """
    Raise ValueError if duplicate IDs are found.
    """
    seen = set()
    duplicates = set()

    for item in items:
        value = getattr(item, attr_name)
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    if duplicates:
        dup_list = ", ".join(sorted(str(x) for x in duplicates)[:10])
        raise ValueError(f"Duplicate values found for '{attr_name}': {dup_list}")


def document_has_text(doc: Document) -> bool:
    if doc.full_text and doc.full_text.strip():
        return True
    if doc.abstract and doc.abstract.strip():
        return True
    return any(sec.section_text.strip() for sec in doc.sections)


def filter_empty_documents(documents: Iterable[Document]) -> List[Document]:
    return [doc for doc in documents if document_has_text(doc)]