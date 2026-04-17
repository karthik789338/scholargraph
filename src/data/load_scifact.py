from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.graph.schemas import Document, DocumentMetadata, GoldEvidence, Query, QueryMetadata, Section
from src.utils.hashing import make_doc_id, make_query_id
from src.utils.io import read_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_scifact(corpus_path: str | Path, claims_path: str | Path) -> tuple[list[dict], list[dict]]:
    corpus = read_jsonl(corpus_path)
    claims = read_jsonl(claims_path)
    logger.info(f"Loaded SciFact corpus: {len(corpus)} docs")
    logger.info(f"Loaded SciFact claims: {len(claims)} claims")
    return corpus, claims


def normalize_scifact_corpus(raw_corpus: List[Dict[str, Any]]) -> List[Document]:
    documents: List[Document] = []

    for item in raw_corpus:
        source_id = str(item["doc_id"])
        doc_id = make_doc_id("scifact", source_id)

        abstract_sentences = item.get("abstract", [])
        abstract_text = " ".join(s.strip() for s in abstract_sentences if s and s.strip())

        sections = []
        if abstract_text:
            sections = [
                Section(
                    section_id=f"{doc_id}_sec_0",
                    section_title="abstract",
                    section_text=abstract_text,
                )
            ]

        doc = Document(
            doc_id=doc_id,
            dataset="scifact",
            title=item.get("title", "").strip(),
            abstract=abstract_text if abstract_text else None,
            full_text=None,
            sections=sections,
            metadata=DocumentMetadata(
                year=item.get("year", None),
                authors=[],
                venue=None,
                domain="scientific",
            ),
        )
        documents.append(doc)

    logger.info(f"Normalized {len(documents)} SciFact documents")
    return documents


def normalize_scifact_claims(raw_claims: List[Dict[str, Any]]) -> List[Query]:
    queries: List[Query] = []

    for item in raw_claims:
        claim_id = str(item["id"])
        cited_docs = item.get("cited_doc_ids", [])
        evidence_map = item.get("evidence", {})

        gold_label = None
        gold_evidence: List[GoldEvidence] = []

        for cited_doc_id, evidence_sets in evidence_map.items():
            doc_id = make_doc_id("scifact", str(cited_doc_id))

            for ev in evidence_sets:
                raw_label = str(ev.get("label", "")).strip().lower()

                if raw_label in {"support", "supports", "entailment"}:
                    gold_label = "supports"
                elif raw_label in {"contradict", "contradiction", "refute", "refutes"}:
                    gold_label = "refutes"

                sent_indices = ev.get("sentences", [])
                sentence_ids = [f"{doc_id}_{doc_id}_sec_0_sent_{idx}" for idx in sent_indices]

                gold_evidence.append(
                    GoldEvidence(
                        doc_id=doc_id,
                        chunk_id=None,
                        sentence_ids=sentence_ids,
                    )
                )

        if gold_label is None:
            gold_label = "insufficient"

        source_doc_id = make_doc_id("scifact", str(cited_docs[0])) if cited_docs else None

        query = Query(
            query_id=make_query_id("scifact", claim_id),
            task_type="claim_verification",
            dataset="scifact",
            doc_scope="closed",
            text=item["claim"].strip(),
            source_doc_id=source_doc_id,
            gold_answer=None,
            gold_label=gold_label,
            gold_evidence=gold_evidence,
            metadata=QueryMetadata(
                question_type=None,
                is_unanswerable=(gold_label == "insufficient"),
                candidate_doc_ids=[make_doc_id("scifact", str(doc_id)) for doc_id in cited_docs],
            ),
        )
        queries.append(query)

    logger.info(f"Normalized {len(queries)} SciFact queries")
    return queries