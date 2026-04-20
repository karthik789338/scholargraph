from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


LABEL_ALIASES = {
    "supports": "supports",
    "support": "supports",
    "supported": "supports",
    "entails": "supports",
    "entailment": "supports",

    "refutes": "refutes",
    "refute": "refutes",
    "refuted": "refutes",
    "contradicts": "refutes",
    "contradiction": "refutes",

    "insufficient": "insufficient",
    "insuff": "insufficient",
    "not enough information": "insufficient",
    "noinfo": "insufficient",
    "no info": "insufficient",
    "neutral": "insufficient",
    "unknown": "insufficient",
}


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_scifact_prompt(
    claim_text: str,
    candidate_chunks: List[Dict[str, Any]],
    max_chunk_chars: int = 900,
) -> str:
    lines: List[str] = []
    lines.append("You are given a scientific claim and several retrieved evidence chunks from scientific papers.")
    lines.append("")
    lines.append("Task:")
    lines.append("1. Decide whether the evidence SUPPORTS, REFUTES, or is INSUFFICIENT for the claim.")
    lines.append("2. Select the chunk numbers that best justify the decision.")
    lines.append("3. If the evidence is insufficient, return no chunks.")
    lines.append("")
    lines.append(f"Claim: {clean_text(claim_text)}")
    lines.append("")
    lines.append("Evidence chunks:")

    if not candidate_chunks:
        lines.append("[1] No evidence available.")
    else:
        for i, chunk in enumerate(candidate_chunks, start=1):
            title = clean_text(str(chunk.get("title", ""))) if chunk.get("title") else ""
            section = clean_text(str(chunk.get("section_title", ""))) if chunk.get("section_title") else ""
            text = truncate_text(str(chunk.get("text", "")), max_chars=max_chunk_chars)

            prefix_bits = []
            if title:
                prefix_bits.append(f"Title: {title}")
            if section:
                prefix_bits.append(f"Section: {section}")

            prefix = ""
            if prefix_bits:
                prefix = " | ".join(prefix_bits) + " | "

            lines.append(f"[{i}] {prefix}{text}")

    lines.append("")
    lines.append("Return your answer in exactly this format:")
    lines.append("LABEL: <SUPPORTS or REFUTES or INSUFFICIENT>")
    lines.append("CHUNKS: <comma-separated chunk numbers, or NONE>")

    return "\n".join(lines)


def normalize_label(raw_label: Optional[str]) -> Optional[str]:
    if raw_label is None:
        return None

    label = raw_label.strip().lower()
    label = label.replace("_", " ")
    label = re.sub(r"[^a-z\s]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()

    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]

    # fuzzy fallback
    if "support" in label or "entail" in label:
        return "supports"
    if "refut" in label or "contradict" in label:
        return "refutes"
    if "insufficient" in label or "neutral" in label or "no info" in label or "unknown" in label:
        return "insufficient"

    return None


def extract_label(raw_output: str) -> Optional[str]:
    # Preferred strict parse
    match = re.search(r"label\s*:\s*([A-Za-z _-]+)", raw_output, flags=re.IGNORECASE)
    if match:
        label = normalize_label(match.group(1))
        if label is not None:
            return label

    # Fallback: look anywhere in text
    lowered = raw_output.lower()
    for candidate in [
        "supports",
        "support",
        "refutes",
        "refute",
        "insufficient",
        "neutral",
        "noinfo",
        "no info",
        "not enough information",
    ]:
        if candidate in lowered:
            label = normalize_label(candidate)
            if label is not None:
                return label

    return None


def extract_chunk_numbers(raw_output: str, num_candidates: int) -> List[int]:
    # Try strict CHUNKS line first
    match = re.search(r"chunks\s*:\s*(.+)", raw_output, flags=re.IGNORECASE)
    text = match.group(1).strip() if match else raw_output.strip()

    lowered = text.lower()
    if "none" in lowered:
        return []

    nums = re.findall(r"\b(\d+)\b", text)
    chunk_numbers: List[int] = []
    seen = set()

    for n_str in nums:
        n = int(n_str)
        if 1 <= n <= num_candidates and n not in seen:
            chunk_numbers.append(n)
            seen.add(n)

    return chunk_numbers


def parse_scifact_output(
    raw_output: str,
    num_candidates: int,
) -> Dict[str, Any]:
    label = extract_label(raw_output)
    chunk_numbers = extract_chunk_numbers(raw_output, num_candidates)

    parse_failed = label is None

    return {
        "predicted_label": label,
        "predicted_chunk_numbers": chunk_numbers,
        "parse_failed": parse_failed,
    }


def candidate_sort_key(candidate: Dict[str, Any]) -> Any:
    # Prefer explicit rank if available
    if "rank" in candidate and candidate["rank"] is not None:
        return (0, int(candidate["rank"]))

    # Otherwise sort descending by common score fields
    for key in ["rrf_score", "score", "retrieval_score", "hybrid_score", "bm25_score", "dense_score"]:
        if key in candidate and candidate[key] is not None:
            return (1, -float(candidate[key]))

    return (2, 0)


def resolve_candidate_chunk_id(candidate: Any) -> Optional[str]:
    if isinstance(candidate, str):
        return candidate

    if isinstance(candidate, dict):
        for key in ["chunk_id", "id"]:
            if key in candidate and candidate[key] is not None:
                return str(candidate[key])

    return None


def build_prompt_candidates(
    graph_input: Dict[str, Any],
    chunks_by_id: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    raw_candidates = graph_input.get("candidate_chunks", []) or []
    if not isinstance(raw_candidates, list):
        return []

    # Sort if rank/score is present
    candidates_sorted = sorted(raw_candidates, key=candidate_sort_key)

    prompt_candidates: List[Dict[str, Any]] = []

    for candidate in candidates_sorted[:top_k]:
        chunk_id = resolve_candidate_chunk_id(candidate)
        if chunk_id is None:
            continue

        chunk_obj = chunks_by_id.get(chunk_id)
        if chunk_obj is None:
            continue

        prompt_candidates.append(
            {
                "chunk_id": chunk_id,
                "title": ((chunk_obj.get("metadata") or {}).get("title")) if isinstance(chunk_obj.get("metadata"), dict) else None,
                "section_title": chunk_obj.get("section_title"),
                "text": chunk_obj.get("text", ""),
            }
        )

    return prompt_candidates
