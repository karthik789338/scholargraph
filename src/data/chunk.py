from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.data.sentence_split import split_document_into_sentences
from src.graph.schemas import Chunk, ChunkMetadata, Document, Section, Sentence
from src.utils.hashing import make_chunk_id


def whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def token_count(text: str) -> int:
    return len(whitespace_tokenize(text))


def _group_sentences_by_section(sentences: Sequence[Sentence]) -> Dict[str | None, List[Sentence]]:
    grouped: Dict[str | None, List[Sentence]] = {}
    for sent in sentences:
        grouped.setdefault(sent.section_id, []).append(sent)
    return grouped


def _find_text_span(section_text: str, chunk_text: str, search_start: int = 0) -> tuple[int, int]:
    """
    Best-effort char span lookup inside the source section text.
    """
    idx = section_text.find(chunk_text, search_start)
    if idx == -1:
        idx = section_text.find(chunk_text)
    if idx == -1:
        return 0, len(chunk_text)
    return idx, idx + len(chunk_text)


def _build_chunks_for_section(
    doc: Document,
    section: Section,
    section_sentences: Sequence[Sentence],
    start_chunk_index: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """
    Sentence-aware chunking with overlap.

    Guarantees progress by never allowing full-chunk overlap.
    """
    if not section_sentences:
        return []

    sent_token_counts = [token_count(s.text) for s in section_sentences]
    chunks: List[Chunk] = []

    start = 0
    chunk_index = start_chunk_index
    search_cursor = 0

    while start < len(section_sentences):
        end = start
        current_tokens = 0

        while end < len(section_sentences):
            next_tokens = sent_token_counts[end]

            # Always allow at least one sentence into a chunk.
            if end == start:
                current_tokens += next_tokens
                end += 1
                continue

            if current_tokens + next_tokens > max_tokens:
                break

            current_tokens += next_tokens
            end += 1

        chunk_sentences = section_sentences[start:end]
        chunk_text = " ".join(s.text for s in chunk_sentences).strip()

        if chunk_text:
            char_start, char_end = _find_text_span(section.section_text, chunk_text, search_cursor)
            search_cursor = max(search_cursor, char_end)

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, section.section_id, chunk_index),
                    doc_id=doc.doc_id,
                    dataset=doc.dataset,
                    section_id=section.section_id,
                    section_title=section.section_title,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    sentence_ids=[s.sentence_id for s in chunk_sentences],
                    is_abstract=(section.section_title or "").lower() == "abstract",
                    metadata=ChunkMetadata(
                        title=doc.title,
                        year=doc.metadata.year,
                    ),
                )
            )
            chunk_index += 1

        if end >= len(section_sentences):
            break

        # Compute overlap start.
        # Important: do not allow overlap to go all the way back to `start`,
        # otherwise we could loop forever.
        overlap_start = end
        overlap_accum = 0

        for j in range(end - 1, start, -1):
            overlap_accum += sent_token_counts[j]
            overlap_start = j
            if overlap_accum >= overlap_tokens:
                break

        start = overlap_start

    return chunks


def chunk_document(
    doc: Document,
    sentences: Optional[Sequence[Sentence]] = None,
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> List[Chunk]:
    """
    Chunk a single document.

    Strategy:
    - if sections exist, chunk section by section
    - else create a pseudo-section from abstract or full_text
    - preserve sentence IDs and section metadata
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")

    if sentences is None:
        sentences = split_document_into_sentences(doc)

    if not sentences:
        return []

    section_to_sentences = _group_sentences_by_section(sentences)

    sections: List[Section] = list(doc.sections)

    if not sections:
        if doc.abstract and doc.abstract.strip():
            sections = [
                Section(
                    section_id=f"{doc.doc_id}_abstract",
                    section_title="abstract",
                    section_text=doc.abstract,
                )
            ]
        elif doc.full_text and doc.full_text.strip():
            sections = [
                Section(
                    section_id=f"{doc.doc_id}_fulltext",
                    section_title="full_text",
                    section_text=doc.full_text,
                )
            ]

    chunks: List[Chunk] = []
    next_chunk_index = 0

    for section in sections:
        section_sentences = section_to_sentences.get(section.section_id, [])
        section_chunks = _build_chunks_for_section(
            doc=doc,
            section=section,
            section_sentences=section_sentences,
            start_chunk_index=next_chunk_index,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(section_chunks)
        next_chunk_index += len(section_chunks)

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    max_tokens: int = 220,
    overlap_tokens: int = 40,
) -> List[Chunk]:
    output: List[Chunk] = []
    for doc in documents:
        output.extend(
            chunk_document(
                doc=doc,
                sentences=None,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    return output


def map_sentences_to_chunks(chunks: Sequence[Chunk]) -> Dict[str, List[str]]:
    """
    sentence_id -> list of chunk_ids
    Useful when aligning evidence later.
    """
    mapping: Dict[str, List[str]] = {}
    for chunk in chunks:
        for sentence_id in chunk.sentence_ids:
            mapping.setdefault(sentence_id, []).append(chunk.chunk_id)
    return mapping


def map_chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def map_chunks_by_doc_id(chunks: Sequence[Chunk]) -> Dict[str, List[Chunk]]:
    grouped: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)
    return grouped