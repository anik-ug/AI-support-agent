from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.common import ensure_dir, stable_hash, write_json
from src.config import AppConfig
from src.data_loader import DatasetSchema, infer_bool_series, infer_schema, load_dataset, maybe_parse_timestamp, standardize_text_column

@dataclass(frozen=True)
class BrandSelection:
    brand_value: str
    support_message_count: int
    customer_message_count: int
    conversation_count: int
    note: str

def _normalize_identifier(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)): return ""
    text = str(value).strip()
    if not text: return ""
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return text

def clean_raw_frame(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["text"] = standardize_text_column(cleaned, schema.text_col)
    cleaned = cleaned[cleaned["text"].str.len() > 0].copy()
    cleaned["timestamp"] = maybe_parse_timestamp(cleaned[schema.timestamp_col]) if schema.timestamp_col else pd.NaT
    cleaned["inbound"] = infer_bool_series(cleaned[schema.inbound_col]) if schema.inbound_col else np.nan
    for name, source in [("tweet_id", schema.tweet_id_col), ("conversation_id", schema.conversation_id_col), ("reply_to_tweet_id", schema.reply_to_col), ("user_id", schema.user_col)]:
        cleaned[name] = cleaned[source] if source and source in cleaned.columns else np.nan
    parent_source = None
    for candidate in ["in_response_to_tweet_id", "in_response_to", "parent_tweet_id", "response_tweet_id"]:
        if candidate in df.columns and df[candidate].notna().any():
            parent_source = candidate
            break
    cleaned["parent_tweet_id"] = df[parent_source].map(_normalize_identifier) if parent_source else cleaned["reply_to_tweet_id"].map(_normalize_identifier)
    if "tweet_id" in cleaned and cleaned["tweet_id"].notna().any():
        cleaned["tweet_id"] = cleaned["tweet_id"].map(_normalize_identifier)
        cleaned = cleaned.drop_duplicates(subset=["tweet_id"])
    return cleaned.reset_index(drop=True)

def derive_message_role(df: pd.DataFrame) -> pd.Series:
    if "inbound" in df.columns and df["inbound"].notna().any():
        return pd.Series(np.where(df["inbound"].fillna(False), "customer", "brand"), index=df.index)
    return pd.Series(["customer"] * len(df), index=df.index)

def assign_conversation_ids(df: pd.DataFrame) -> pd.Series:
    if "conversation_id" in df.columns and df["conversation_id"].notna().any():
        return df["conversation_id"].astype(str)
    if "tweet_id" not in df.columns or "parent_tweet_id" not in df.columns:
        return pd.Series([f"conv_{i}" for i in range(len(df))], index=df.index)
    parent = {}
    for row in df[["tweet_id", "parent_tweet_id"]].itertuples(index=False):
        tid = _normalize_identifier(row.tweet_id)
        pid = _normalize_identifier(row.parent_tweet_id)
        if tid: parent[tid] = pid if pid else ""
    memo = {}
    def root(tid: str) -> str:
        if tid in memo: return memo[tid]
        seen = []
        cur = tid
        while cur and cur in parent and parent[cur] and parent[cur] not in seen:
            seen.append(cur)
            cur = parent[cur]
            if cur in memo:
                cur = memo[cur]
                break
        result = cur or tid
        for item in seen: memo[item] = result
        memo[tid] = result
        return result
    return df["tweet_id"].map(lambda x: root(_normalize_identifier(x)))

def derive_brand_column(df: pd.DataFrame) -> pd.Series:
    if "role" not in df.columns or "user_id" not in df.columns:
        return pd.Series(["unknown_brand"] * len(df), index=df.index)
    brand_by_conversation = {}
    for conversation_id, group in df.groupby("conversation_id", dropna=False):
        candidates = group.loc[group["role"] == "brand", "user_id"].astype(str)
        if not candidates.empty:
            brand_by_conversation[str(conversation_id)] = str(candidates.mode().iloc[0])
    return df["conversation_id"].astype(str).map(brand_by_conversation).fillna(df["user_id"].astype(str))

def summarize_brand_candidates(df: pd.DataFrame) -> pd.DataFrame:
    support = df[df["role"] == "brand"].copy()
    if support.empty:
        return pd.DataFrame(columns=["brand_id", "customer_messages", "brand_messages", "conversations"])
    support_counts = support.groupby("user_id").agg(brand_messages=("tweet_id", "count"), conversations=("conversation_id", "nunique")).reset_index().rename(columns={"user_id": "brand_id"})
    customer_counts = df[df["role"] == "customer"].groupby("conversation_id").size()
    support_counts["customer_messages"] = support_counts["conversations"].map(lambda n: 0)
    for i, row in support_counts.iterrows():
        convs = set(support.loc[support["user_id"].astype(str) == str(row["brand_id"]), "conversation_id"].astype(str))
        support_counts.loc[i, "customer_messages"] = int(df[df["conversation_id"].astype(str).isin(convs) & (df["role"] == "customer")].shape[0])
    return support_counts[["brand_id", "customer_messages", "brand_messages", "conversations"]].sort_values(["brand_messages", "customer_messages", "conversations"], ascending=False).reset_index(drop=True)

def choose_brand(df: pd.DataFrame, requested_brand: str | None = None) -> BrandSelection:
    summary = summarize_brand_candidates(df)
    if requested_brand:
        match = summary[summary["brand_id"].astype(str) == str(requested_brand)]
        if match.empty: raise ValueError(f"BRAND_ID={requested_brand!r} was not found among outbound support authors.")
        best = match.iloc[0]
        note = "Selected from BRAND_ID in .env."
    else:
        if summary.empty: raise ValueError("No outbound support-account candidates found.")
        best = summary.iloc[0]
        note = "Automatically selected the highest-volume outbound support author. Set BRAND_ID in .env to pin a brand."
    return BrandSelection(str(best["brand_id"]), int(best["brand_messages"]), int(best["customer_messages"]), int(best["conversations"]), note)

def filter_brand_thread(df: pd.DataFrame, brand_value: str) -> pd.DataFrame:
    brand_value = str(brand_value)
    support_conversations = set(df.loc[(df["role"] == "brand") & (df["user_id"].astype(str) == brand_value), "conversation_id"].astype(str))
    filtered = df[df["conversation_id"].astype(str).isin(support_conversations)].copy()
    filtered["brand_id"] = brand_value
    filtered["message_hash"] = filtered["text"].map(stable_hash)
    return filtered.sort_values(["conversation_id", "timestamp", "tweet_id"], na_position="last").reset_index(drop=True)

def build_customer_brand_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for conversation_id, group in df.groupby("conversation_id", dropna=False):
        ordered = group.sort_values(["timestamp", "tweet_id"], na_position="last").reset_index(drop=True)
        previous_customer = None
        for row in ordered.itertuples(index=False):
            if row.role == "customer":
                previous_customer = row
            elif row.role == "brand" and previous_customer is not None:
                customer_message = str(previous_customer.text)
                brand_reply = str(row.text)
                rows.append({
                    "conversation_id": str(conversation_id),
                    "brand_id": str(row.brand_id),
                    "customer_message": customer_message,
                    "brand_reply": brand_reply,
                    "customer_message_hash": stable_hash(customer_message),
                    "brand_reply_hash": stable_hash(brand_reply),
                    "customer_count": 1,
                    "brand_count": 1,
                    "timestamp": getattr(row, "timestamp", pd.NaT),
                })
                previous_customer = None
    return pd.DataFrame(rows)

def sample_brand_data(df: pd.DataFrame, *, sample_size: int, seed: int) -> pd.DataFrame:
    if df.empty: return df
    conversations = df["conversation_id"].astype(str).unique().tolist()
    if sample_size > 0 and len(conversations) > sample_size:
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(conversations, size=sample_size, replace=False).tolist())
        return df[df["conversation_id"].astype(str).isin(chosen)].copy()
    return df.copy()

def prepare_brand_dataset(config: AppConfig, raw_df: pd.DataFrame | None = None):
    path = config.resolved_dataset_path()
    if raw_df is None:
        raw_df = load_dataset(path)
    schema = infer_schema(raw_df)
    cleaned = clean_raw_frame(raw_df, schema)
    cleaned["conversation_id"] = assign_conversation_ids(cleaned)
    cleaned["role"] = derive_message_role(cleaned)
    selection = choose_brand(cleaned, config.brand_id)
    brand_df = filter_brand_thread(cleaned, selection.brand_value)
    brand_df = sample_brand_data(brand_df, sample_size=config.sample_size, seed=config.random_seed)
    return brand_df, selection, schema

def save_processed_artifacts(config: AppConfig, brand_df: pd.DataFrame, selection: BrandSelection, schema: DatasetSchema) -> None:
    ensure_dir(config.processed_data_dir)
    brand_df.to_csv(config.processed_data_dir / "brand_conversations.csv", index=False)
    write_json(config.processed_data_dir / "brand_selection.json", selection)
    write_json(config.processed_data_dir / "schema.json", schema.__dict__)

def build_golden_eval_template(pairs: pd.DataFrame, path: Path, *, sample_size: int = 200, seed: int = 13) -> pd.DataFrame:
    ensure_dir(path.parent)
    if pairs.empty:
        out = pd.DataFrame(columns=["message", "intent", "expected_action", "difficulty", "notes"])
    else:
        candidates = pairs.drop_duplicates(subset=["customer_message_hash"]).copy()
        if len(candidates) > sample_size:
            candidates = candidates.sample(n=sample_size, random_state=seed)
        out = pd.DataFrame({"message": candidates["customer_message"].astype(str), "intent": "", "expected_action": "", "difficulty": "", "notes": ""})
    out.to_csv(path, index=False)
    return out

def split_train_eval(pairs: pd.DataFrame, eval_ratio: float = 0.2, seed: int = 13):
    if pairs.empty: return pairs.copy(), pairs.copy()
    convs = pairs["conversation_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(seed); rng.shuffle(convs)
    eval_ids = set(convs[:max(1, int(len(convs) * eval_ratio))])
    eval_df = pairs[pairs["conversation_id"].astype(str).isin(eval_ids)].copy()
    train_df = pairs[~pairs["conversation_id"].astype(str).isin(eval_ids)].copy()
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)
