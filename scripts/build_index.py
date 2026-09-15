from __future__ import annotations

from pathlib import Path

import pandas as pd

from _bootstrap import add_project_root_to_path

add_project_root_to_path()

from src.config import AppConfig, ensure_workspace_dirs
from src.retriever import SentenceTransformerRetriever


def main() -> None:
    config = AppConfig.from_env()
    ensure_workspace_dirs()
    train_path = config.processed_data_dir / "train_pairs.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run scripts/prepare_data.py first.")
    train_df = pd.read_csv(train_path)
    retriever = SentenceTransformerRetriever(config.embedding_model).fit(train_df)
    retriever.save(config.processed_data_dir / "retriever")
    print(f"Indexed {len(train_df):,} historical pairs.")


if __name__ == "__main__":
    main()
