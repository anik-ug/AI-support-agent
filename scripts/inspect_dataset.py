from __future__ import annotations
import argparse
import pandas as pd
from _bootstrap import add_project_root_to_path
add_project_root_to_path()
from src.config import AppConfig, ensure_workspace_dirs
from src.data_loader import load_dataset, infer_schema
from src.data_processing import clean_raw_frame, derive_message_role, assign_conversation_ids, summarize_brand_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    config = AppConfig.from_env(); ensure_workspace_dirs()
    path = config.resolved_dataset_path()
    print(f"Dataset: {path}")
    print(f"Size: {path.stat().st_size / (1024**2):.1f} MB")
    df = load_dataset(path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    schema = infer_schema(df)
    cleaned = clean_raw_frame(df, schema)
    cleaned["conversation_id"] = assign_conversation_ids(cleaned)
    cleaned["role"] = derive_message_role(cleaned)
    summary = summarize_brand_candidates(cleaned)
    print("\nTop outbound support accounts:")
    print(summary.head(args.top).to_string(index=False))
    print("\nRecommendation: put the desired brand_id from this table into BRAND_ID in .env.")

if __name__ == "__main__": main()
