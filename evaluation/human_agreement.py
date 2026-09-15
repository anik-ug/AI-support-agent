from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.common import ensure_dir


HUMAN_TEMPLATE_COLUMNS = ["message", "generated_reply", "relevance_human", "helpfulness_human", "groundedness_human", "brand_consistency_human", "safety_human"]


def create_human_rating_template(output_path: Path, examples: pd.DataFrame) -> pd.DataFrame:
    template = examples.copy()
    for column in HUMAN_TEMPLATE_COLUMNS[2:]:
        template[column] = ""
    template = template[HUMAN_TEMPLATE_COLUMNS]
    ensure_dir(output_path.parent)
    template.to_csv(output_path, index=False)
    return template


def compute_human_agreement(human_df: pd.DataFrame, judge_df: pd.DataFrame) -> dict:
    merged = human_df.merge(judge_df, on=["message", "generated_reply"], how="inner", suffixes=("_human", "_judge"))
    results = {}
    for metric in ["relevance", "helpfulness", "groundedness", "brand_consistency", "safety"]:
        human_column = f"{metric}_human"
        judge_column = f"{metric}_judge"
        if human_column not in merged.columns or judge_column not in merged.columns or merged.empty:
            results[metric] = {"spearman": None, "status": "pending"}
            continue
        rho = spearmanr(merged[human_column], merged[judge_column], nan_policy="omit").correlation
        results[metric] = {"spearman": None if rho is None else float(rho), "status": "complete"}
    return results
