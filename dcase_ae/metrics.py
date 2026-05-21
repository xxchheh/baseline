from dataclasses import dataclass
from math import isnan

import numpy as np


METRIC_HEADER = [
    "AUC (source)",
    "AUC (target)",
    "pAUC",
    "pAUC (source)",
    "pAUC (target)",
    "precision (source)",
    "precision (target)",
    "recall (source)",
    "recall (target)",
    "F1 score (source)",
    "F1 score (target)",
]


@dataclass(frozen=True)
class ScoreRecord:
    filename: str
    score: float
    label: int | None
    domain: str | None


def parse_label_from_filename(filename: str) -> int | None:
    lower_name = filename.lower()
    if "anomaly" in lower_name:
        return 1
    if "normal" in lower_name:
        return 0
    return None


def parse_domain_from_filename(filename: str) -> str | None:
    lower_name = filename.lower()
    if "source" in lower_name:
        return "source"
    if "target" in lower_name:
        return "target"
    return None


def compute_metrics(records: list[ScoreRecord], max_fpr: float, threshold: float | None) -> dict[str, float]:
    labeled_records = [record for record in records if record.label is not None]
    if not labeled_records:
        return {name: float("nan") for name in METRIC_HEADER}

    threshold = _select_threshold(labeled_records, threshold)
    y_true = np.asarray([record.label for record in labeled_records], dtype=int)
    y_score = np.asarray([record.score for record in labeled_records], dtype=float)

    metrics = {
        "AUC (source)": _domain_auc(labeled_records, "source"),
        "AUC (target)": _domain_auc(labeled_records, "target"),
        "pAUC": _safe_roc_auc(y_true, y_score, max_fpr=max_fpr),
        "pAUC (source)": _domain_pauc(labeled_records, "source", max_fpr),
        "pAUC (target)": _domain_pauc(labeled_records, "target", max_fpr),
    }
    for domain in ["source", "target"]:
        precision, recall, f1 = _classification_metrics(labeled_records, domain, threshold)
        metrics[f"precision ({domain})"] = precision
        metrics[f"recall ({domain})"] = recall
        metrics[f"F1 score ({domain})"] = f1
    return metrics


def format_metric(value: float) -> str:
    if value is None or isnan(value):
        return "nan"
    return f"{value:.10f}"


def _select_threshold(records: list[ScoreRecord], threshold: float | None) -> float:
    if threshold is not None:
        return threshold
    normal_scores = [record.score for record in records if record.label == 0]
    if normal_scores:
        return float(np.quantile(normal_scores, 0.9))
    return float(np.quantile([record.score for record in records], 0.9))


def _domain_auc(records: list[ScoreRecord], domain: str) -> float:
    selected = [
        record
        for record in records
        if record.domain == domain or record.label == 1
    ]
    return _records_auc(selected)


def _domain_pauc(records: list[ScoreRecord], domain: str, max_fpr: float) -> float:
    selected = [record for record in records if record.domain == domain]
    return _records_auc(selected, max_fpr=max_fpr)


def _records_auc(records: list[ScoreRecord], max_fpr: float | None = None) -> float:
    if not records:
        return float("nan")
    y_true = np.asarray([record.label for record in records], dtype=int)
    y_score = np.asarray([record.score for record in records], dtype=float)
    return _safe_roc_auc(y_true, y_score, max_fpr=max_fpr)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float | None = None) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = 1 + threshold_indices - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    positives = tps[-1]
    negatives = fps[-1]
    if positives == 0 or negatives == 0:
        return float("nan")

    tpr = tps / positives
    fpr = fps / negatives
    if max_fpr is None or max_fpr >= 1.0:
        return _trapezoid_area(tpr, fpr)
    if max_fpr <= 0:
        return float("nan")

    stop = np.searchsorted(fpr, max_fpr, side="right")
    x_interp = [max_fpr]
    y_interp = [np.interp(max_fpr, fpr, tpr)]
    partial_fpr = np.append(fpr[:stop], x_interp)
    partial_tpr = np.append(tpr[:stop], y_interp)
    partial_auc = _trapezoid_area(partial_tpr, partial_fpr)
    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    return float(0.5 * (1.0 + (partial_auc - min_area) / (max_area - min_area)))


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def _classification_metrics(
    records: list[ScoreRecord],
    domain: str,
    threshold: float,
) -> tuple[float, float, float]:
    selected = [record for record in records if record.domain == domain]
    if not selected:
        return float("nan"), float("nan"), float("nan")

    y_true = np.asarray([record.label for record in selected], dtype=int)
    y_pred = np.asarray([record.score > threshold for record in selected], dtype=bool)
    tp = int(np.sum((y_true == 1) & y_pred))
    fp = int(np.sum((y_true == 0) & y_pred))
    fn = int(np.sum((y_true == 1) & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
