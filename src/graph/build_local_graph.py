from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.graph.schemas import (
    Chunk,
    ClaimNode,
    EvidenceNode,
    GraphEdge,
    GraphInput,
)
from src.utils.hashing import make_claim_id, make_edge_id
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_graph_inputs(path: str | Path) -> List[GraphInput]:
    records = read_jsonl(path)
    graph_inputs = [GraphInput(**record) for record in records]
    logger.info(f"Loaded {len(graph_inputs)} graph inputs from {path}")
    return graph_inputs


def load_chunks(path: str | Path) -> List[Chunk]:
    records = read_jsonl(path)
    chunks = [Chunk(**record) for record in records]
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def map_chunks_by_id(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in chunks}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split()).strip()


def token_overlap_score(a: str | None, b: str | None) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())

    if not ta or not tb:
        return 0.0

    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0

    return inter / union


def build_claim_nodes(graph_input: GraphInput) -> List[ClaimNode]:
    """
    If candidate_claims are present, use them.
    Otherwise create a pseudo-claim from the query text.
    """
    claim_nodes: List[ClaimNode] = []

    if graph_input.candidate_claims:
        for idx, claim_dict in enumerate(graph_input.candidate_claims):
            claim_id = claim_dict.get("claim_id") or make_claim_id(graph_input.query_id, str(idx))
            claim_nodes.append(
                ClaimNode(
                    claim_id=claim_id,
                    query_id=graph_input.query_id,
                    doc_id=claim_dict.get("doc_id"),
                    text=claim_dict.get("text", graph_input.query_text),
                    source=claim_dict.get("source", "extracted"),
                    claim_type=claim_dict.get("claim_type", "unknown"),
                    confidence=float(claim_dict.get("confidence", 0.0)),
                )
            )
        return claim_nodes

    claim_nodes.append(
        ClaimNode(
            claim_id=make_claim_id(graph_input.query_id, "query"),
            query_id=graph_input.query_id,
            doc_id=None,
            text=graph_input.query_text,
            source="extracted",
            claim_type="unknown",
            confidence=1.0,
        )
    )
    return claim_nodes


def build_evidence_nodes(
    graph_input: GraphInput,
    chunks_by_id: Dict[str, Chunk],
) -> List[EvidenceNode]:
    evidence_nodes: List[EvidenceNode] = []
    retrieval_scores = graph_input.metadata.get("retrieval_scores", {})

    for chunk_id in graph_input.candidate_chunks:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        score_payload = retrieval_scores.get(chunk_id, {})
        evidence_nodes.append(
            EvidenceNode(
                evidence_id=f"evidence::{chunk.chunk_id}",
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source="gold" if chunk.chunk_id in graph_input.gold_evidence_chunks else "retrieved",
                relevance_score=float(score_payload.get("score", 0.0)),
            )
        )

    return evidence_nodes


def infer_claim_evidence_relation(
    graph_input: GraphInput,
    claim_text: str,
    evidence_node: EvidenceNode,
) -> tuple[str, float]:
    """
    Simple MVP relation inference.

    Priority:
    1. gold evidence + gold label
    2. QA gold evidence => supports
    3. lexical overlap heuristic => related / mentions
    """
    if evidence_node.chunk_id in set(graph_input.gold_evidence_chunks):
        if graph_input.task_type == "claim_verification":
            if graph_input.gold_label in {"supports", "refutes"}:
                return graph_input.gold_label, max(0.8, evidence_node.relevance_score)
            if graph_input.gold_label == "insufficient":
                return "insufficient", max(0.6, evidence_node.relevance_score)
        return "supports", max(0.8, evidence_node.relevance_score)

    overlap = token_overlap_score(claim_text, evidence_node.text)

    if overlap >= 0.18:
        return "related", max(overlap, evidence_node.relevance_score)
    if overlap >= 0.08:
        return "mentions", max(overlap, evidence_node.relevance_score)

    return "related", max(0.05, min(0.15, evidence_node.relevance_score))


def build_claim_to_evidence_edges(
    graph_input: GraphInput,
    claim_nodes: Sequence[ClaimNode],
    evidence_nodes: Sequence[EvidenceNode],
) -> List[GraphEdge]:
    edges: List[GraphEdge] = []

    for claim in claim_nodes:
        for evidence in evidence_nodes:
            relation, score = infer_claim_evidence_relation(
                graph_input=graph_input,
                claim_text=claim.text,
                evidence_node=evidence,
            )

            edges.append(
                GraphEdge(
                    edge_id=make_edge_id(claim.claim_id, evidence.evidence_id, relation),
                    src_id=claim.claim_id,
                    dst_id=evidence.evidence_id,
                    src_type="claim",
                    dst_type="evidence",
                    relation=relation,
                    score=float(score),
                )
            )

    return edges


def build_evidence_to_evidence_edges(
    evidence_nodes: Sequence[EvidenceNode],
    chunks_by_id: Dict[str, Chunk],
) -> List[GraphEdge]:
    """
    Add lightweight structure among evidence nodes:
    - related if same document
    - slightly stronger if adjacent chunk indices within same document
    """
    edges: List[GraphEdge] = []

    for i in range(len(evidence_nodes)):
        for j in range(i + 1, len(evidence_nodes)):
            a = evidence_nodes[i]
            b = evidence_nodes[j]

            chunk_a = chunks_by_id.get(a.chunk_id)
            chunk_b = chunks_by_id.get(b.chunk_id)

            if chunk_a is None or chunk_b is None:
                continue

            if a.doc_id != b.doc_id:
                continue

            score = 0.3
            if abs(chunk_a.chunk_index - chunk_b.chunk_index) <= 1:
                score = 0.5

            relation = "related"
            edge_id = make_edge_id(a.evidence_id, b.evidence_id, relation)

            edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    src_id=a.evidence_id,
                    dst_id=b.evidence_id,
                    src_type="evidence",
                    dst_type="evidence",
                    relation=relation,
                    score=score,
                )
            )

    return edges


def build_local_graph(
    graph_input: GraphInput,
    chunks_by_id: Dict[str, Chunk],
) -> Dict[str, Any]:
    claim_nodes = build_claim_nodes(graph_input)
    evidence_nodes = build_evidence_nodes(graph_input, chunks_by_id)

    claim_edges = build_claim_to_evidence_edges(
        graph_input=graph_input,
        claim_nodes=claim_nodes,
        evidence_nodes=evidence_nodes,
    )
    evidence_edges = build_evidence_to_evidence_edges(
        evidence_nodes=evidence_nodes,
        chunks_by_id=chunks_by_id,
    )

    graph_dict = {
        "query_id": graph_input.query_id,
        "task_type": graph_input.task_type,
        "query_text": graph_input.query_text,
        "claims": [node.model_dump() for node in claim_nodes],
        "evidence": [node.model_dump() for node in evidence_nodes],
        "edges": [edge.model_dump() for edge in claim_edges + evidence_edges],
        "metadata": {
            **graph_input.metadata,
            "gold_evidence_chunks": graph_input.gold_evidence_chunks,
            "gold_label": graph_input.gold_label,
            "num_claim_nodes": len(claim_nodes),
            "num_evidence_nodes": len(evidence_nodes),
            "num_edges": len(claim_edges) + len(evidence_edges),
        },
    }
    return graph_dict


def build_local_graphs(
    graph_inputs: Sequence[GraphInput],
    chunks_by_id: Dict[str, Chunk],
) -> List[Dict[str, Any]]:
    local_graphs: List[Dict[str, Any]] = []

    for idx, graph_input in enumerate(graph_inputs, start=1):
        local_graphs.append(build_local_graph(graph_input, chunks_by_id))

        if idx % 100 == 0:
            logger.info(f"Built local graphs for {idx}/{len(graph_inputs)} queries")

    logger.info(f"Built {len(local_graphs)} local graphs")
    return local_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local claim-evidence graphs from graph inputs.")
    parser.add_argument("--graph-inputs", required=True, help="Path to graph_inputs JSONL")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")
    parser.add_argument("--output", required=True, help="Path to output local_graphs JSONL")

    args = parser.parse_args()

    graph_inputs = load_graph_inputs(args.graph_inputs)
    chunks = load_chunks(args.chunks)
    chunks_by_id = map_chunks_by_id(chunks)

    local_graphs = build_local_graphs(
        graph_inputs=graph_inputs,
        chunks_by_id=chunks_by_id,
    )

    write_jsonl(local_graphs, args.output)
    logger.info(f"Wrote local graphs to {args.output}")


if __name__ == "__main__":
    main()