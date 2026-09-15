from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.common import ensure_dir, write_json


@dataclass
class IntentEvaluationResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, Any]
    confusion_matrix: list[list[int]]


def evaluate_intent_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> IntentEvaluationResult:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    return IntentEvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        per_class=report,
        confusion_matrix=cm.tolist(),
    )


def save_intent_results(result: IntentEvaluationResult, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    write_json(output_path, result)
