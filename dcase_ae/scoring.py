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


@dataclass(frozen=True)
class HealthResult:
    health_score: float
    health_level: str


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


def health_score_from_reference(
    raw_score: float,
    normal_reference_scores: np.ndarray,
) -> HealthResult:
    reference = np.asarray(normal_reference_scores, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        raise ValueError("normal_reference_scores is empty.")

    p50, p90, p95, p99 = np.quantile(reference, [0.50, 0.90, 0.95, 0.99])
    raw_score = float(raw_score)
    if raw_score <= p50:
        health_score = 100.0
    elif raw_score <= p90:
        health_score = _linear_map(raw_score, p50, p90, 100.0, 95.0)
    elif raw_score <= p95:
        health_score = _linear_map(raw_score, p90, p95, 95.0, 90.0)
    elif raw_score <= p99:
        health_score = _linear_map(raw_score, p95, p99, 90.0, 40.0)
    else:
        scale = max(float(p99 - p95), np.finfo(float).eps)
        health_score = 40.0 - 40.0 * ((raw_score - p99) / scale)

    health_score = float(np.clip(health_score, 0.0, 100.0))
    return HealthResult(
        health_score=health_score,
        health_level=classify_health(health_score),
    )


def classify_health(health_score: float) -> str:
    if health_score >= 90.0:
        return "healthy"
    if health_score >= 80.0:
        return "slightly_degraded"
    if health_score >= 60.0:
        return "warning"
    if health_score >= 20.0:
        return "unhealthy"
    return "critical"


def _linear_map(
    value: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    if x1 <= x0:
        return y1
    ratio = (value - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


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
