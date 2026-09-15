from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.common import ensure_dir, write_json


def evaluate_escalation(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["AUTO_HANDLE", "ESCALATE"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0, average="binary", pos_label="ESCALATE")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


def save_escalation_results(result: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    write_json(output_path, result)
