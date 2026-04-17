from __future__ import annotations

import hashlib


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_doc_id(dataset: str, source_id: str) -> str:
    return f"{dataset}_doc_{source_id}"


def make_section_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_sec_{idx}"


def make_sentence_id(doc_id: str, section_id: str | None, sent_idx: int) -> str:
    sid = section_id if section_id else "abstract"
    return f"{doc_id}_{sid}_sent_{sent_idx}"


def make_chunk_id(doc_id: str, section_id: str | None, chunk_idx: int) -> str:
    sid = section_id if section_id else "abstract"
    return f"{doc_id}_{sid}_chunk_{chunk_idx}"


def make_query_id(dataset: str, source_id: str) -> str:
    return f"{dataset}_query_{source_id}"


def make_claim_id(query_id: str, suffix: str = "0") -> str:
    return f"{query_id}_claim_{suffix}"


def make_edge_id(src_id: str, dst_id: str, relation: str) -> str:
    return stable_hash(f"{src_id}|{dst_id}|{relation}", length=16)