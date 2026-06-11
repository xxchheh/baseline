from dataclasses import dataclass

import numpy as np


RISK_THRESHOLDS = {
    "normal_like": 90.0,
    "suspicious": 95.0,
    "abnormal": 99.0,
}


@dataclass(frozen=True)
class AnomalyResult:
    raw_score: float
    abnormality_score: float
    risk_level: str


def score_to_percentile(
    raw_score: float,
    normal_reference_scores: np.ndarray,
) -> float:
    reference = np.asarray(normal_reference_scores, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        raise ValueError("normal_reference_scores is empty.")
    return float(100.0 * np.mean(reference <= raw_score))


def classify_risk(abnormality_score: float) -> str:
    if abnormality_score < RISK_THRESHOLDS["normal_like"]:
        return "normal_like"
    if abnormality_score < RISK_THRESHOLDS["suspicious"]:
        return "suspicious"
    if abnormality_score < RISK_THRESHOLDS["abnormal"]:
        return "abnormal"
    return "highly_abnormal"


def anomaly_result(
    raw_score: float,
    normal_reference_scores: np.ndarray,
) -> AnomalyResult:
    abnormality_score = score_to_percentile(raw_score, normal_reference_scores)
    return AnomalyResult(
        raw_score=float(raw_score),
        abnormality_score=abnormality_score,
        risk_level=classify_risk(abnormality_score),
    )
