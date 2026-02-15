"""Bridge to nanda-model-integrity-layer.

Demonstrates embedding governance approval data into the integrity
layer's provenance and agent facts systems.

Requires: pip install nanda-model-governance[integrity]

Usage::

    python examples/with_integrity_layer.py
"""

from __future__ import annotations

from nanda_governance import GovernanceCoordinator, ModelApproval
from nanda_governance.nanda import (
    approval_to_integrity_facts,
    create_provenance_with_approval,
)


def main() -> None:
    coord = GovernanceCoordinator()

    # Create a governance approval
    output = coord.complete_training(
        model_id="classifier-v4",
        weights_hash="sha256:112233445566",
        metrics={"loss": 0.18},
    )
    approval = coord.submit_for_governance(
        output, approved_by="governance-lead"
    )

    # Convert to integrity facts
    facts = approval_to_integrity_facts(approval)
    print("Governance facts for agent registry:")
    for key, value in facts["governance"].items():
        print(f"  {key}: {value}")

    # Create provenance with governance metadata
    try:
        provenance = create_provenance_with_approval(
            approval,
            model_name="classifier-v4",
            model_version="4.0.0",
        )
        print(f"\nProvenance created: {provenance}")
    except ImportError:
        print(
            "\nnanda-model-integrity-layer not installed. "
            "Install with: pip install nanda-model-governance[integrity]"
        )


if __name__ == "__main__":
    main()
