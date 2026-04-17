from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    rows: List[dict] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "model_dump"):
                obj = record.model_dump()
            elif hasattr(record, "dict"):
                obj = record.dict()
            else:
                obj = record
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")