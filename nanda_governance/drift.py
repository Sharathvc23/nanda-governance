"""Drift detection for deployed models.

Compares serving-time metrics against training baselines to detect
model degradation.  Default behavior is alert-only — no automatic
revocation without explicit opt-in.

Zero external dependencies: uses a custom KS statistic approximation
instead of scipy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nanda_governance._types import DriftSeverity

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection thresholds.

    Attributes:
        max_loss_increase: Maximum relative loss increase before drift.
        min_accuracy_ratio: Minimum serving/training accuracy ratio.
        confidence_threshold: Minimum confidence to trigger a drift alert.
        ks_threshold: KS statistic threshold for distribution drift.
    """

    max_loss_increase: float = 0.20
    min_accuracy_ratio: float = 0.95
    confidence_threshold: float = 0.90
    ks_threshold: float = 0.10


DEFAULT_DRIFT_CONFIG = DriftConfig()


@dataclass
class DriftMetric:
    """A single metric comparison result."""

    name: str
    training_value: float
    serving_value: float
    threshold: float
    is_drifted: bool
    drift_severity: float = 0.0
    details: str = ""


@dataclass
class DriftCheckResult:
    """Result of a drift check operation."""

    model_id: str
    checked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_drifted: bool = False
    metrics: list[DriftMetric] = field(default_factory=list)
    overall_severity: float = 0.0
    confidence: float = 0.0
    summary: str = ""
    recommended_action: str = "monitor"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for evidence logging."""
        return {
            "model_id": self.model_id,
            "checked_at": self.checked_at.isoformat(),
            "is_drifted": self.is_drifted,
            "metrics": [
                {
                    "name": m.name,
                    "training_value": m.training_value,
                    "serving_value": m.serving_value,
                    "threshold": m.threshold,
                    "is_drifted": m.is_drifted,
                    "drift_severity": m.drift_severity,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "overall_severity": self.overall_severity,
            "confidence": self.confidence,
            "summary": self.summary,
            "recommended_action": self.recommended_action,
        }


@dataclass
class DriftAlert:
    """Alert event emitted when drift is detected."""

    model_id: str
    severity: str
    result: DriftCheckResult
    alerted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for notification systems."""
        return {
            "model_id": self.model_id,
            "severity": self.severity,
            "alerted_at": self.alerted_at.isoformat(),
            "summary": self.result.summary,
            "recommended_action": self.result.recommended_action,
            "overall_severity": self.result.overall_severity,
            "confidence": self.result.confidence,
        }


# ---------------------------------------------------------------------------
# Core drift-check logic
# ---------------------------------------------------------------------------


def check_drift(
    model_id: str,
    training_metrics: dict[str, float],
    serving_metrics: dict[str, float],
    *,
    config: DriftConfig | None = None,
) -> DriftCheckResult:
    """Check for drift between training and serving metrics.

    Args:
        model_id: The model being checked.
        training_metrics: Baseline metrics from training.
        serving_metrics: Current metrics from production.
        config: Drift detection configuration.

    Returns:
        DriftCheckResult with drift assessment.
    """
    if config is None:
        config = DEFAULT_DRIFT_CONFIG

    metrics: list[DriftMetric] = []
    drifted_count = 0
    total_severity = 0.0

    # Check loss metric
    if "loss" in training_metrics and "loss" in serving_metrics:
        training_loss = training_metrics["loss"]
        serving_loss = serving_metrics["loss"]

        if training_loss > 0:
            loss_increase = (serving_loss - training_loss) / training_loss
            is_drifted = loss_increase > config.max_loss_increase
            severity = (
                min(1.0, loss_increase / config.max_loss_increase)
                if config.max_loss_increase > 0
                else 0.0
            )

            metrics.append(
                DriftMetric(
                    name="loss",
                    training_value=training_loss,
                    serving_value=serving_loss,
                    threshold=config.max_loss_increase,
                    is_drifted=is_drifted,
                    drift_severity=severity if is_drifted else 0.0,
                    details=f"Loss increased by {loss_increase:.1%}",
                )
            )

            if is_drifted:
                drifted_count += 1
                total_severity += severity

    # Check accuracy metric
    if "accuracy" in training_metrics and "accuracy" in serving_metrics:
        training_acc = training_metrics["accuracy"]
        serving_acc = serving_metrics["accuracy"]

        if training_acc > 0:
            acc_ratio = serving_acc / training_acc
            is_drifted = acc_ratio < config.min_accuracy_ratio
            severity = (
                min(
                    1.0,
                    (config.min_accuracy_ratio - acc_ratio)
                    / config.min_accuracy_ratio,
                )
                if is_drifted
                else 0.0
            )

            metrics.append(
                DriftMetric(
                    name="accuracy",
                    training_value=training_acc,
                    serving_value=serving_acc,
                    threshold=config.min_accuracy_ratio,
                    is_drifted=is_drifted,
                    drift_severity=severity,
                    details=f"Accuracy ratio is {acc_ratio:.2%}",
                )
            )

            if is_drifted:
                drifted_count += 1
                total_severity += severity

    # Aggregate assessment
    overall_drifted = drifted_count > 0
    overall_severity = total_severity / max(len(metrics), 1)
    confidence = 1.0 if len(metrics) > 0 else 0.0

    if not overall_drifted:
        recommended_action = "monitor"
        summary = "No drift detected"
    elif overall_severity < 0.3:
        recommended_action = "monitor"
        summary = f"Minor drift detected ({drifted_count} metrics)"
    elif overall_severity < 0.6:
        recommended_action = "investigate"
        summary = f"Moderate drift detected ({drifted_count} metrics)"
    elif overall_severity < 0.8:
        recommended_action = "investigate"
        summary = f"Significant drift detected ({drifted_count} metrics)"
    else:
        recommended_action = "consider_revoke"
        summary = f"Severe drift detected ({drifted_count} metrics)"

    return DriftCheckResult(
        model_id=model_id,
        is_drifted=overall_drifted
        and confidence >= config.confidence_threshold,
        metrics=metrics,
        overall_severity=overall_severity,
        confidence=confidence,
        summary=summary,
        recommended_action=recommended_action,
    )


def check_distribution_drift(
    model_id: str,
    training_outputs: list[float],
    serving_outputs: list[float],
    *,
    config: DriftConfig | None = None,
) -> DriftCheckResult:
    """Check for distribution drift using a custom KS statistic.

    Uses a two-sample KS approximation without requiring scipy.

    Args:
        model_id: The model being checked.
        training_outputs: Sample of outputs from training evaluation.
        serving_outputs: Sample of outputs from production serving.
        config: Drift detection configuration.

    Returns:
        DriftCheckResult with distribution drift assessment.
    """
    if config is None:
        config = DEFAULT_DRIFT_CONFIG

    if len(training_outputs) < 10 or len(serving_outputs) < 10:
        return DriftCheckResult(
            model_id=model_id,
            is_drifted=False,
            summary="Insufficient samples for distribution drift check",
            recommended_action="monitor",
            confidence=0.0,
        )

    # Normalize to [0, 1] range
    all_values = training_outputs + serving_outputs
    min_val, max_val = min(all_values), max(all_values)
    range_val = max_val - min_val if max_val > min_val else 1.0

    training_norm = sorted(
        (v - min_val) / range_val for v in training_outputs
    )
    serving_norm = sorted(
        (v - min_val) / range_val for v in serving_outputs
    )

    # Approximate KS statistic: max |CDF_A - CDF_B|
    n_train, n_serve = len(training_norm), len(serving_norm)
    max_diff = 0.0

    i, j = 0, 0
    while i < n_train and j < n_serve:
        if training_norm[i] <= serving_norm[j]:
            diff = abs((i + 1) / n_train - j / n_serve)
            i += 1
        else:
            diff = abs(i / n_train - (j + 1) / n_serve)
            j += 1
        max_diff = max(max_diff, diff)

    ks_statistic = max_diff
    is_drifted = ks_statistic > config.ks_threshold

    return DriftCheckResult(
        model_id=model_id,
        is_drifted=is_drifted,
        metrics=[
            DriftMetric(
                name="ks_statistic",
                training_value=0.0,
                serving_value=ks_statistic,
                threshold=config.ks_threshold,
                is_drifted=is_drifted,
                drift_severity=(
                    min(1.0, ks_statistic / config.ks_threshold)
                    if is_drifted
                    else 0.0
                ),
                details=f"KS statistic: {ks_statistic:.3f}",
            )
        ],
        overall_severity=(
            min(1.0, ks_statistic / config.ks_threshold)
            if is_drifted
            else 0.0
        ),
        confidence=0.95,
        summary=(
            f"Distribution drift "
            f"{'detected' if is_drifted else 'not detected'} "
            f"(KS={ks_statistic:.3f})"
        ),
        recommended_action="investigate" if is_drifted else "monitor",
    )


def create_drift_alert(
    result: DriftCheckResult,
) -> DriftAlert | None:
    """Create an alert from a drift check result if drift was detected.

    Returns None if no drift was detected.
    """
    if not result.is_drifted:
        return None

    if result.overall_severity >= 0.8:
        severity = DriftSeverity.CRITICAL.value
    elif result.overall_severity >= 0.6:
        severity = DriftSeverity.HIGH.value
    elif result.overall_severity >= 0.3:
        severity = DriftSeverity.MEDIUM.value
    else:
        severity = DriftSeverity.LOW.value

    return DriftAlert(
        model_id=result.model_id,
        severity=severity,
        result=result,
    )


__all__ = [
    "DEFAULT_DRIFT_CONFIG",
    "DriftAlert",
    "DriftCheckResult",
    "DriftConfig",
    "DriftMetric",
    "check_distribution_drift",
    "check_drift",
    "create_drift_alert",
]
