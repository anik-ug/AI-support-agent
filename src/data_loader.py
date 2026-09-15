from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
from src.common import normalize_whitespace

TWCS_COLUMNS = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".feather", ".json", ".jsonl"}

@dataclass(frozen=True)
class DatasetSchema:
    text_col: str
    user_col: str | None
    brand_col: str | None
    inbound_col: str | None
    timestamp_col: str | None
    conversation_id_col: str | None
    tweet_id_col: str | None
    reply_to_col: str | None

def discover_raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists(): return []
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)

def load_raw_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv": return pd.read_csv(path)
    if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
    if path.suffix.lower() == ".feather": return pd.read_feather(path)
    if path.suffix.lower() == ".jsonl": return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json": return pd.read_json(path)
    raise ValueError(f"Unsupported raw file type: {path}")

def load_raw_dataset(raw_dir: Path) -> pd.DataFrame:
    files = discover_raw_files(raw_dir)
    if not files: return pd.DataFrame()
    frames = []
    for file_path in files:
        frame = load_raw_file(file_path).copy()
        frame["source_file"] = file_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)

def load_dataset(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv": return pd.read_csv(path, usecols=usecols)
    return load_raw_file(path)

def iter_csv(path: Path, *, chunksize: int = 100_000, usecols: list[str] | None = None):
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize)

def _normalize_columns(columns: Iterable[str]) -> dict[str, str]:
    return {col.lower().replace(" ", "_").replace("-", "_"): col for col in columns}

def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = _normalize_columns(df.columns)
    for candidate in candidates:
        normalized = candidate.lower().replace(" ", "_").replace("-", "_")
        if normalized in lookup: return lookup[normalized]
    return None

def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    if df.empty: raise ValueError("Cannot infer schema from an empty dataframe.")
    text_col = _pick_column(df, ["text", "message", "tweet_text", "content", "body", "clean_text"])
    if text_col is None: raise ValueError(f"Could not find a text column in columns: {list(df.columns)}")
    return DatasetSchema(
        text_col=text_col,
        user_col=_pick_column(df, ["user", "username", "author", "author_id", "author_handle", "account", "handle"]),
        brand_col=_pick_column(df, ["brand", "company", "brand_name", "support_account", "brand_handle", "account_name"]),
        inbound_col=_pick_column(df, ["inbound", "is_customer", "customer_message", "from_customer", "direction"]),
        timestamp_col=_pick_column(df, ["created_at", "timestamp", "time", "date", "datetime"]),
        conversation_id_col=_pick_column(df, ["conversation_id", "thread_id", "conversation", "thread", "dialogue_id"]),
        tweet_id_col=_pick_column(df, ["tweet_id", "id", "message_id", "status_id"]),
        reply_to_col=_pick_column(df, ["response_tweet_id", "in_response_to_tweet_id", "reply_to_tweet_id", "parent_tweet_id", "response_to", "response_tweet_id"]),
    )

def standardize_text_column(df: pd.DataFrame, text_col: str) -> pd.Series:
    return df[text_col].fillna("").astype(str).map(normalize_whitespace)

def infer_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool: return series.fillna(False)
    return series.fillna("").astype(str).str.lower().str.strip().isin({"1", "true", "t", "yes", "y", "customer", "inbound"})

def maybe_parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)
