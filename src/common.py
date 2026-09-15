from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    if is_dataclass(data):
        data = asdict(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_json_block(text: str) -> str | None:
    match = JSON_RE.search(text)
    if match:
        return match.group(1)
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped
    return None


def safe_json_loads(text: str) -> Any:
    block = extract_json_block(text)
    if block is None:
        raise ValueError("No JSON object found in model response.")
    return json.loads(block)


def chunked(values: Iterable[Any], size: int) -> list[list[Any]]:
    batch: list[Any] = []
    batches: list[list[Any]] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
