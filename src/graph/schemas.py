from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DatasetName = Literal["qasper", "scifact", "scifact_open", "s2orc"]
TaskType = Literal["qa", "claim_verification"]
ClaimType = Literal["finding", "comparison", "method", "limitation", "hypothesis", "unknown"]
RelationType = Literal["supports", "refutes", "mentions", "related", "insufficient"]


class Section(BaseModel):
    section_id: str
    section_title: Optional[str] = None
    section_text: str


class DocumentMetadata(BaseModel):
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    domain: Optional[str] = None


class Document(BaseModel):
    doc_id: str
    dataset: DatasetName
    title: str
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    sections: List[Section] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class Sentence(BaseModel):
    sentence_id: str
    doc_id: str
    section_id: Optional[str] = None
    sentence_index: int
    text: str


class ChunkMetadata(BaseModel):
    title: Optional[str] = None
    year: Optional[int] = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    dataset: DatasetName
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    chunk_index: int
    text: str
    char_start: int = 0
    char_end: int = 0
    sentence_ids: List[str] = Field(default_factory=list)
    is_abstract: bool = False
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class GoldEvidence(BaseModel):
    doc_id: str
    chunk_id: Optional[str] = None
    sentence_ids: List[str] = Field(default_factory=list)


class QueryMetadata(BaseModel):
    question_type: Optional[str] = None
    is_unanswerable: bool = False
    candidate_doc_ids: List[str] = Field(default_factory=list)


class Query(BaseModel):
    query_id: str
    task_type: TaskType
    dataset: DatasetName
    doc_scope: Literal["closed", "open"] = "closed"
    text: str
    source_doc_id: Optional[str] = None
    gold_answer: Optional[str] = None
    gold_label: Optional[Literal["supports", "refutes", "insufficient"]] = None
    gold_evidence: List[GoldEvidence] = Field(default_factory=list)
    metadata: QueryMetadata = Field(default_factory=QueryMetadata)


class ClaimNode(BaseModel):
    claim_id: str
    query_id: Optional[str] = None
    doc_id: Optional[str] = None
    text: str
    source: Literal["dataset_gold", "extracted"]
    claim_type: ClaimType = "unknown"
    confidence: float = 0.0


class EvidenceNode(BaseModel):
    evidence_id: str
    doc_id: str
    chunk_id: str
    text: str
    source: Literal["gold", "retrieved", "predicted"]
    relevance_score: float = 0.0


class GraphEdge(BaseModel):
    edge_id: str
    src_id: str
    dst_id: str
    src_type: Literal["claim", "evidence"]
    dst_type: Literal["claim", "evidence"]
    relation: RelationType
    score: float = 0.0


class GraphInput(BaseModel):
    query_id: str
    task_type: TaskType
    query_text: str
    candidate_chunks: List[str] = Field(default_factory=list)
    gold_evidence_chunks: List[str] = Field(default_factory=list)
    gold_label: Optional[Literal["supports", "refutes", "insufficient"]] = None
    candidate_claims: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)