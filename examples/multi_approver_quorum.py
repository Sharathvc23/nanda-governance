"""Multi-approver quorum with Ed25519 signing.

Demonstrates requiring two governance approvers to sign off before
a model can be deployed.

Requires: pip install nanda-model-governance[crypto]

Usage::

    python examples/multi_approver_quorum.py
"""

from __future__ import annotations

import asyncio

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from nanda_governance import GovernanceCoordinator


async def main() -> None:
    # Generate keys for two approvers
    key_alice = Ed25519PrivateKey.generate()
    key_bob = Ed25519PrivateKey.generate()
    pub_alice = key_alice.public_key()

    coord = GovernanceCoordinator()

    # Training
    output = coord.complete_training(
        model_id="classifier-v2",
        weights_hash="sha256:fedcba0987654321",
        metrics={"loss": 0.22, "accuracy": 0.96},
    )

    # First approval (Alice) — requires 2 signatures
    approval = coord.submit_for_governance(
        output,
        approved_by="alice",
        private_key=key_alice,
        required_approvers=2,
    )
    print(f"Alice signed. Quorum met? {approval.has_quorum()}")

    # Second approval (Bob)
    has_quorum = coord.add_approval_signature(
        "classifier-v2", "bob", key_bob
    )
    print(f"Bob signed. Quorum met? {has_quorum}")

    # Deploy with signature verification
    result = await coord.deploy_approved(
        approval, public_key=pub_alice
    )
    print(f"Deployed: {result.promoted}")


if __name__ == "__main__":
    asyncio.run(main())
