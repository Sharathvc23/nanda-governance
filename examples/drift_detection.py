"""Drift detection example.

Shows how to monitor deployed models for drift and optionally
auto-revoke on severe degradation.

Usage::

    python examples/drift_detection.py
"""

from __future__ import annotations

from nanda_governance import (
    GovernanceCoordinator,
    DriftConfig,
    check_distribution_drift,
    create_drift_alert,
)


def main() -> None:
    coord = GovernanceCoordinator()

    # Set up a model
    output = coord.complete_training(
        model_id="recommender-v1",
        weights_hash="sha256:aabbccdd",
        metrics={"loss": 0.25, "accuracy": 0.93},
    )
    coord.submit_for_governance(output, approved_by="alice")

    # Scenario 1: minor drift — no action needed
    print("--- Scenario 1: Minor drift ---")
    result = coord.check_drift(
        "recommender-v1",
        training_metrics={"loss": 0.25, "accuracy": 0.93},
        serving_metrics={"loss": 0.28, "accuracy": 0.91},
    )
    print(f"Drifted: {result.is_drifted}, Action: {result.recommended_action}")

    # Scenario 2: severe drift with auto-revoke
    print("\n--- Scenario 2: Severe drift (auto-revoke) ---")
    result = coord.check_drift(
        "recommender-v1",
        training_metrics={"loss": 0.25},
        serving_metrics={"loss": 2.50},
        auto_revoke=True,
    )
    print(f"Drifted: {result.is_drifted}, Action: {result.recommended_action}")
    print(f"Still approved? {coord.store.is_approved('recommender-v1')}")

    # Scenario 3: distribution drift with custom config
    print("\n--- Scenario 3: Distribution drift ---")
    import random
    random.seed(42)
    training_outputs = [random.gauss(0.5, 0.1) for _ in range(200)]
    serving_outputs = [random.gauss(0.7, 0.15) for _ in range(200)]

    dist_result = check_distribution_drift(
        "recommender-v1",
        training_outputs,
        serving_outputs,
        config=DriftConfig(ks_threshold=0.05),
    )
    print(f"Distribution drift: {dist_result.is_drifted}")
    print(f"Summary: {dist_result.summary}")

    alert = create_drift_alert(dist_result)
    if alert:
        print(f"Alert severity: {alert.severity}")


if __name__ == "__main__":
    main()
