from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.intent_classifier import MajorityIntentClassifier, TfidfIntentClassifier


@dataclass
class BaselineArtifacts:
    majority: MajorityIntentClassifier
    tfidf: TfidfIntentClassifier


def train_majority_baseline(train_df: pd.DataFrame, text_col: str = "message", label_col: str = "intent") -> MajorityIntentClassifier:
    model = MajorityIntentClassifier()
    model.fit(train_df[text_col].astype(str).tolist(), train_df[label_col].astype(str).tolist())
    return model


def train_tfidf_baseline(train_df: pd.DataFrame, text_col: str = "message", label_col: str = "intent") -> TfidfIntentClassifier:
    model = TfidfIntentClassifier()
    model.fit(train_df[text_col].astype(str).tolist(), train_df[label_col].astype(str).tolist())
    return model
