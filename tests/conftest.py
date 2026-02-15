"""Shared fixtures for nanda_governance tests."""

from __future__ import annotations

import pytest

from nanda_governance.approval import ModelApproval
from nanda_governance.stores.memory import InMemoryApprovalStore


@pytest.fixture
def store() -> InMemoryApprovalStore:
    """Fresh in-memory approval store."""
    return InMemoryApprovalStore()


@pytest.fixture
def sample_approval() -> ModelApproval:
    """A minimal active approval with quorum=1."""
    approval = ModelApproval(
        model_id="model-abc",
        weights_hash="sha256:deadbeef" * 4,
        approved_by="governance-alice",
        profile="default",
        correlation_id="corr-001",
    )
    approval.add_signature("governance-alice", "sig-placeholder")
    return approval


HAS_CRYPTOGRAPHY = False
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    HAS_CRYPTOGRAPHY = True
except ImportError:
    pass

skip_no_crypto = pytest.mark.skipif(
    not HAS_CRYPTOGRAPHY,
    reason="cryptography package not installed",
)
