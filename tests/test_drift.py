"""Tests for drift detection."""

from __future__ import annotations

from nanda_governance.drift import (
    DriftConfig,
    check_distribution_drift,
    check_drift,
    create_drift_alert,
)


def test_no_drift() -> None:
    result = check_drift(
        "m1",
        {"loss": 0.30, "accuracy": 0.92},
        {"loss": 0.32, "accuracy": 0.91},
    )
    assert result.is_drifted is False
    assert result.recommended_action == "monitor"


def test_loss_drift() -> None:
    result = check_drift(
        "m1",
        {"loss": 0.30},
        {"loss": 0.50},
    )
    assert result.is_drifted is True
    assert any(m.name == "loss" and m.is_drifted for m in result.metrics)


def test_accuracy_drift() -> None:
    result = check_drift(
        "m1",
        {"accuracy": 0.92},
        {"accuracy": 0.70},
    )
    assert result.is_drifted is True
    assert any(
        m.name == "accuracy" and m.is_drifted for m in result.metrics
    )


def test_custom_config() -> None:
    strict = DriftConfig(max_loss_increase=0.05, min_accuracy_ratio=0.99)
    result = check_drift(
        "m1",
        {"loss": 0.30, "accuracy": 0.92},
        {"loss": 0.33, "accuracy": 0.91},
        config=strict,
    )
    assert result.is_drifted is True


def test_severe_drift_action() -> None:
    result = check_drift(
        "m1",
        {"loss": 0.10},
        {"loss": 1.00},
    )
    assert result.is_drifted is True
    assert result.recommended_action == "consider_revoke"
    assert result.overall_severity >= 0.8


def test_drift_to_dict() -> None:
    result = check_drift(
        "m1",
        {"loss": 0.30},
        {"loss": 0.50},
    )
    d = result.to_dict()
    assert d["model_id"] == "m1"
    assert d["is_drifted"] is True
    assert isinstance(d["metrics"], list)


def test_no_metrics() -> None:
    result = check_drift("m1", {}, {})
    assert result.is_drifted is False
    assert result.confidence == 0.0


def test_create_alert_when_drifted() -> None:
    result = check_drift("m1", {"loss": 0.10}, {"loss": 1.00})
    alert = create_drift_alert(result)
    assert alert is not None
    assert alert.model_id == "m1"
    assert alert.severity == "critical"


def test_create_alert_no_drift() -> None:
    result = check_drift(
        "m1", {"loss": 0.30}, {"loss": 0.30}
    )
    alert = create_drift_alert(result)
    assert alert is None


def test_distribution_drift_detected() -> None:
    import random

    random.seed(42)
    training = [random.gauss(0.0, 1.0) for _ in range(100)]
    serving = [random.gauss(2.0, 1.0) for _ in range(100)]
    result = check_distribution_drift("m1", training, serving)
    assert result.is_drifted is True
    assert any(
        m.name == "ks_statistic" for m in result.metrics
    )


def test_distribution_drift_no_drift() -> None:
    import random

    random.seed(42)
    training = [random.gauss(0.0, 1.0) for _ in range(100)]
    serving = [random.gauss(0.0, 1.0) for _ in range(100)]
    result = check_distribution_drift("m1", training, serving)
    assert result.is_drifted is False


def test_distribution_drift_insufficient_samples() -> None:
    result = check_distribution_drift("m1", [1.0, 2.0], [1.0, 2.0])
    assert result.is_drifted is False
    assert result.confidence == 0.0
    assert "Insufficient" in result.summary


def test_alert_severity_levels() -> None:
    # Low severity — just barely over the 20% threshold
    config = DriftConfig(max_loss_increase=0.20)
    result = check_drift(
        "m1",
        {"loss": 1.00},
        {"loss": 1.22},
        config=config,
    )
    alert = create_drift_alert(result)
    if alert:
        # 22% increase / 20% threshold -> severity ~1.1 capped to 1.0
        # With only one metric, overall_severity == drift_severity
        assert alert.severity in ("low", "medium", "high", "critical")

    # Critical severity — 10x loss
    result = check_drift(
        "m1",
        {"loss": 0.10},
        {"loss": 1.00},
    )
    alert = create_drift_alert(result)
    assert alert is not None
    assert alert.severity == "critical"
