from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.graph.schemas import Document, DocumentMetadata, GoldEvidence, Query, QueryMetadata
from src.utils.hashing import make_doc_id, make_query_id
from src.utils.io import read_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_scifact_open_claims(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    claims = read_jsonl(path)
    logger.info(f"Loaded {len(claims)} SciFact-Open claims from {path}")
    return claims


def load_scifact_open_documents(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    docs = read_jsonl(path)
    logger.info(f"Loaded {len(docs)} SciFact-Open documents from {path}")
    return docs


def normalize_scifact_open_documents(raw_docs: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []

    for item in raw_docs:
        pmid = str(item["pmid"])
        doc_id = make_doc_id("scifact_open", pmid)

        documents.append(
            Document(
                doc_id=doc_id,
                dataset="scifact_open",
                title=item.get("title", "") or "",
                abstract=item.get("abstract", "") or None,
                full_text=item.get("abstract", "") or None,
                sections=[],
                metadata=DocumentMetadata(
                    authors=[],
                    year=item.get("year"),
                    venue=item.get("journal"),
                    domain="scientific",
                ),
            )
        )

    logger.info(f"Normalized {len(documents)} SciFact-Open documents")
    return documents


def normalize_scifact_open_queries(raw_claims: List[Dict[str, Any]]) -> List[Query]:
    queries: List[Query] = []

    for item in raw_claims:
        claim_id = str(item["id"])
        claim_text = str(item["claim"]).strip()
        evidence = item.get("evidence", {})

        gold_label = "insufficient"
        gold_evidence: List[GoldEvidence] = []

        if isinstance(evidence, dict):
            for raw_doc_id, ann in evidence.items():
                if ann is None:
                    continue
                if not isinstance(ann, dict):
                    continue

                doc_id = make_doc_id("scifact_open", str(raw_doc_id))
                label = ann.get("label", "")
                sentences = ann.get("sentences", []) or []

                if label == "SUPPORT":
                    gold_label = "supports"
                elif label == "CONTRADICT":
                    gold_label = "refutes"

                sentence_ids = [
                    f"{doc_id}_{doc_id}_sec_0_sent_{idx}"
                    for idx in sentences
                ]

                gold_evidence.append(
                    GoldEvidence(
                        doc_id=doc_id,
                        chunk_id=None,
                        sentence_ids=sentence_ids,
                    )
                )

        query = Query(
            query_id=make_query_id("scifact_open", claim_id),
            task_type="claim_verification",
            dataset="scifact_open",
            doc_scope="open",
            text=claim_text,
            source_doc_id=None,
            gold_answer=None,
            gold_label=gold_label,
            gold_evidence=gold_evidence,
            metadata=QueryMetadata(
                question_type=None,
                is_unanswerable=(gold_label == "insufficient"),
            ),
        )
        queries.append(query)

    logger.info(f"Normalized {len(queries)} SciFact-Open queries")
    return queries

def load_scifact_open_full_corpus(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    docs = read_jsonl(path)
    logger.info(f"Loaded {len(docs)} SciFact-Open full-corpus docs from {path}")
    return docs


def normalize_scifact_open_full_documents(raw_docs: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []

    for item in raw_docs:
        raw_id = str(item["_id"])
        doc_id = make_doc_id("scifact_open", raw_id)

        title = item.get("title", "") or ""
        text = item.get("text", "") or ""

        documents.append(
            Document(
                doc_id=doc_id,
                dataset="scifact_open",
                title=title,
                abstract=text if text else None,
                full_text=text if text else None,
                sections=[],
                metadata=DocumentMetadata(
                    authors=[],
                    year=None,
                    venue=None,
                    domain="scientific",
                ),
            )
        )

    logger.info(f"Normalized {len(documents)} SciFact-Open full-corpus documents")
    return documents