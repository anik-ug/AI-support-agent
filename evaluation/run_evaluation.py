from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.config import AppConfig, ensure_workspace_dirs
from src.common import read_json, write_json
from src.intent_classifier import TfidfIntentClassifier
from evaluation.evaluate_intent import evaluate_intent_predictions

def main():
    config = AppConfig.from_env(); ensure_workspace_dirs()
    golden = pd.read_csv(config.golden_eval_path)
    required = {"message", "intent", "expected_action"}
    missing = required - set(golden.columns)
    if missing: raise ValueError(f"Golden set missing columns: {sorted(missing)}")
    if golden.empty or golden["intent"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Golden set is not fully labeled. Fill data/golden_eval.csv before evaluation.")
    train = pd.read_csv(config.processed_data_dir / "train_pairs.csv")
    if "intent" not in train.columns:
        raise RuntimeError("train_pairs.csv has no intent labels. Intent labels must be assigned before training the supervised baseline.")
    train = train.dropna(subset=["intent"])
    if train["intent"].nunique() < 2:
        raise RuntimeError("Need at least two labeled intents in training data for the TF-IDF baseline.")
    model = TfidfIntentClassifier().fit(train["customer_message"].astype(str).tolist(), train["intent"].astype(str).tolist())
    predictions = model.predict(golden["message"].astype(str).tolist())
    result = evaluate_intent_predictions(golden["intent"].astype(str).tolist(), [p.intent for p in predictions], sorted(golden["intent"].astype(str).unique()))
    write_json(config.results_dir / "intent_tfidf.json", result)
    print(json.dumps(result.__dict__, indent=2))

if __name__ == "__main__": main()
