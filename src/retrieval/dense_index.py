from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.graph.schemas import Chunk
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, object] = {}


@dataclass
class DenseIndex:
    embeddings: np.ndarray
    chunks: List[Chunk]
    model_name: str
    normalized: bool = True


def load_chunks(path: str | Path) -> List[Chunk]:
    records = read_jsonl(path)
    chunks = [Chunk(**record) for record in records]
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def _load_encoder(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install it with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def build_dense_index(
    chunks: Sequence[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> DenseIndex:
    model = _load_encoder(model_name)
    texts = [chunk.text for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    if not normalize_embeddings:
        embeddings = _normalize_rows(embeddings)

    logger.info(
        f"Built dense index over {len(chunks)} chunks with model {model_name}"
    )

    return DenseIndex(
        embeddings=embeddings.astype(np.float32),
        chunks=list(chunks),
        model_name=model_name,
        normalized=True,
    )


def save_dense_index(index: DenseIndex, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)

    embeddings_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.json"
    chunks_path = output_dir / "chunks.jsonl"

    np.save(embeddings_path, index.embeddings)

    write_json(
        {
            "num_chunks": len(index.chunks),
            "index_type": "dense",
            "model_name": index.model_name,
            "normalized": index.normalized,
            "embedding_dim": int(index.embeddings.shape[1]),
        },
        meta_path,
    )

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in index.chunks:
            f.write(chunk.model_dump_json() + "\n")

    logger.info(f"Saved dense index to {output_dir}")


def load_dense_index(index_dir: str | Path) -> DenseIndex:
    index_dir = Path(index_dir)

    embeddings = np.load(index_dir / "embeddings.npy")
    meta = read_json(index_dir / "metadata.json")
    chunks = load_chunks(index_dir / "chunks.jsonl")

    index = DenseIndex(
        embeddings=embeddings.astype(np.float32),
        chunks=chunks,
        model_name=meta["model_name"],
        normalized=bool(meta.get("normalized", True)),
    )

    logger.info(f"Loaded dense index from {index_dir}")
    return index


def encode_queries(
    queries: Sequence[str],
    model_name: str,
    batch_size: int = 64,
) -> np.ndarray:
    model = _load_encoder(model_name)
    query_embeddings = model.encode(
        list(queries),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return query_embeddings.astype(np.float32)


def _format_ranked_results(
    ranked: Sequence[tuple[float, Chunk]],
) -> List[dict]:
    output = []
    for rank, (score, chunk) in enumerate(ranked, start=1):
        output.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            }
        )
    return output


def search_dense_with_embedding(
    index: DenseIndex,
    query_embedding: np.ndarray,
    top_k: int = 10,
    allowed_doc_ids: Optional[set[str]] = None,
) -> List[dict]:
    scores = index.embeddings @ query_embedding

    candidate_indices = list(range(len(index.chunks)))
    if allowed_doc_ids is not None:
        candidate_indices = [
            idx for idx, chunk in enumerate(index.chunks)
            if chunk.doc_id in allowed_doc_ids
        ]

    ranked = sorted(
        ((float(scores[idx]), index.chunks[idx]) for idx in candidate_indices),
        key=lambda x: x[0],
        reverse=True,
    )[:top_k]

    return _format_ranked_results(ranked)


def batch_search_dense(
    index: DenseIndex,
    queries: Sequence[str],
    top_k: int = 10,
    allowed_doc_ids_per_query: Optional[Sequence[Optional[set[str]]]] = None,
    model_name_override: Optional[str] = None,
    batch_size: int = 64,
) -> List[List[dict]]:
    model_name = model_name_override or index.model_name
    query_embeddings = encode_queries(
        queries=queries,
        model_name=model_name,
        batch_size=batch_size,
    )

    if allowed_doc_ids_per_query is None:
        allowed_doc_ids_per_query = [None] * len(queries)

    all_results: List[List[dict]] = []
    for query_embedding, allowed_doc_ids in zip(query_embeddings, allowed_doc_ids_per_query):
        results = search_dense_with_embedding(
            index=index,
            query_embedding=query_embedding,
            top_k=top_k,
            allowed_doc_ids=allowed_doc_ids,
        )
        all_results.append(results)

    return all_results


def search_dense(
    index: DenseIndex,
    query: str,
    top_k: int = 10,
    doc_id: Optional[str] = None,
    model_name_override: Optional[str] = None,
) -> List[dict]:
    model_name = model_name_override or index.model_name
    query_embedding = encode_queries([query], model_name=model_name, batch_size=1)[0]

    allowed_doc_ids = {doc_id} if doc_id is not None else None

    return search_dense_with_embedding(
        index=index,
        query_embedding=query_embedding,
        top_k=top_k,
        allowed_doc_ids=allowed_doc_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense retrieval index from chunk JSONL.")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save dense index")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    index = build_dense_index(
        chunks=chunks,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    save_dense_index(index, args.output_dir)


if __name__ == "__main__":
    main()