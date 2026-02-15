"""Basic approval flow — training -> governance -> serving.

Demonstrates the simplest possible three-plane handoff without
cryptographic signing.

Usage::

    python examples/basic_approval_flow.py
"""

from __future__ import annotations

import asyncio

from nanda_governance import GovernanceCoordinator


async def main() -> None:
    coord = GovernanceCoordinator()

    # 1. Training Plane — produce a handoff object
    output = coord.complete_training(
        model_id="sentiment-v3",
        weights_hash="sha256:abcdef1234567890",
        metrics={"loss": 0.28, "accuracy": 0.94},
        card={"description": "Sentiment classifier v3"},
    )
    print(f"Training complete: {output.model_id}")

    # 2. Governance Plane — create approval
    approval = coord.submit_for_governance(
        output,
        approved_by="governance-team",
        approved_environments=["staging", "production"],
        approval_ttl_days=90,
    )
    print(f"Approved: {approval.approval_id} (expires {approval.expires_at})")

    # 3. Serving Plane — deploy
    result = await coord.deploy_approved(
        approval, environment="staging"
    )
    print(f"Deployed: {result.promoted} at {result.promoted_at}")


if __name__ == "__main__":
    asyncio.run(main())
