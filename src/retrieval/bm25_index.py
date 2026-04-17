from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from src.graph.schemas import Chunk
from src.utils.io import ensure_dir, read_jsonl, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: List[Chunk]
    tokenized_corpus: List[List[str]]
    chunk_id_to_idx: Dict[str, int]


def load_chunks(path: str | Path) -> List[Chunk]:
    records = read_jsonl(path)
    chunks = [Chunk(**record) for record in records]
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_bm25_index(chunks: Sequence[Chunk]) -> BM25Index:
    tokenized_corpus = [simple_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(chunks)}

    logger.info(f"Built BM25 index over {len(chunks)} chunks")
    return BM25Index(
        bm25=bm25,
        chunks=list(chunks),
        tokenized_corpus=tokenized_corpus,
        chunk_id_to_idx=chunk_id_to_idx,
    )


def save_bm25_index(index: BM25Index, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)

    bm25_path = output_dir / "bm25.pkl"
    meta_path = output_dir / "metadata.json"
    chunks_path = output_dir / "chunks.jsonl"

    with bm25_path.open("wb") as f:
        pickle.dump(
            {
                "bm25": index.bm25,
                "tokenized_corpus": index.tokenized_corpus,
                "chunk_id_to_idx": index.chunk_id_to_idx,
            },
            f,
        )

    write_json(
        {
            "num_chunks": len(index.chunks),
            "index_type": "bm25",
            "tokenizer": "simple_tokenize",
        },
        meta_path,
    )

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in index.chunks:
            f.write(chunk.model_dump_json() + "\n")

    logger.info(f"Saved BM25 index to {output_dir}")


def load_bm25_index(index_dir: str | Path) -> BM25Index:
    index_dir = Path(index_dir)

    bm25_path = index_dir / "bm25.pkl"
    chunks_path = index_dir / "chunks.jsonl"

    with bm25_path.open("rb") as f:
        payload = pickle.load(f)

    chunks = load_chunks(chunks_path)

    index = BM25Index(
        bm25=payload["bm25"],
        chunks=chunks,
        tokenized_corpus=payload["tokenized_corpus"],
        chunk_id_to_idx=payload["chunk_id_to_idx"],
    )

    logger.info(f"Loaded BM25 index from {index_dir}")
    return index


def search_bm25(
    index: BM25Index,
    query: str,
    top_k: int = 10,
    doc_id: Optional[str] = None,
) -> List[dict]:
    query_tokens = simple_tokenize(query)
    scores = index.bm25.get_scores(query_tokens)

    results: List[tuple[float, Chunk]] = []
    for score, chunk in zip(scores, index.chunks):
        if doc_id is not None and chunk.doc_id != doc_id:
            continue
        results.append((float(score), chunk))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    output = []
    for rank, (score, chunk) in enumerate(top_results, start=1):
        output.append(
            {
                "rank": rank,
                "score": score,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BM25 index from chunk JSONL.")
    parser.add_argument("--chunks", required=True, help="Path to chunk JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save BM25 index")

    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    index = build_bm25_index(chunks)
    save_bm25_index(index, args.output_dir)


if __name__ == "__main__":
    main()