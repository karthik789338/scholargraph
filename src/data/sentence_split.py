from __future__ import annotations

from typing import Iterable, List, Optional

import spacy

from src.graph.schemas import Document, Section, Sentence
from src.utils.hashing import make_sentence_id


def _build_sentencizer():
    """
    Uses a lightweight spaCy pipeline that does not require downloading
    a large language model.
    """
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


_NLP = _build_sentencizer()


def split_text_into_sentences(text: str) -> List[str]:
    """
    Split raw text into sentence strings.
    """
    if not text or not text.strip():
        return []

    doc = _NLP(text)
    sentences = [span.text.strip() for span in doc.sents if span.text and span.text.strip()]
    return sentences


def _split_section_to_sentences(
    doc_id: str,
    section: Section,
) -> List[Sentence]:
    if not section.section_text.strip():
        return []

    doc = _NLP(section.section_text)
    output: List[Sentence] = []

    for sent_idx, span in enumerate(doc.sents):
        sent_text = span.text.strip()
        if not sent_text:
            continue

        output.append(
            Sentence(
                sentence_id=make_sentence_id(doc_id, section.section_id, sent_idx),
                doc_id=doc_id,
                section_id=section.section_id,
                sentence_index=sent_idx,
                text=sent_text,
            )
        )

    return output


def split_document_into_sentences(doc: Document) -> List[Sentence]:
    """
    Split a Document into sentence objects.

    Preference order:
    1. sections
    2. abstract as a pseudo-section
    3. full_text as a fallback pseudo-section
    """
    all_sentences: List[Sentence] = []

    if doc.sections:
        for section in doc.sections:
            all_sentences.extend(_split_section_to_sentences(doc.doc_id, section))
        return all_sentences

    if doc.abstract and doc.abstract.strip():
        pseudo_section = Section(
            section_id=f"{doc.doc_id}_abstract",
            section_title="abstract",
            section_text=doc.abstract,
        )
        return _split_section_to_sentences(doc.doc_id, pseudo_section)

    if doc.full_text and doc.full_text.strip():
        pseudo_section = Section(
            section_id=f"{doc.doc_id}_fulltext",
            section_title="full_text",
            section_text=doc.full_text,
        )
        return _split_section_to_sentences(doc.doc_id, pseudo_section)

    return all_sentences


def split_documents_into_sentences(documents: Iterable[Document]) -> List[Sentence]:
    output: List[Sentence] = []
    for doc in documents:
        output.extend(split_document_into_sentences(doc))
    return output


def get_section_sentences(
    sentences: Iterable[Sentence],
    doc_id: str,
    section_id: Optional[str],
) -> List[Sentence]:
    return [
        s
        for s in sentences
        if s.doc_id == doc_id and s.section_id == section_id
    ]