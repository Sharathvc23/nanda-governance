"""Tests for GovernanceCoordinator — full three-plane flow."""

from __future__ import annotations

from typing import Any

import pytest

from nanda_governance.coordinator import GovernanceCoordinator
from nanda_governance.contracts import ValidationResult
from nanda_governance.stores.memory import InMemoryApprovalStore
from tests.conftest import HAS_CRYPTOGRAPHY, skip_no_crypto

if HAS_CRYPTOGRAPHY:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class FakeLedger:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, entry_type: str, data: dict[str, Any]) -> str:
        self.entries.append({"type": entry_type, "data": data})
        return f"entry-{len(self.entries)}"


class FakeEndpoint:
    def __init__(self) -> None:
        self.deployed: list[str] = []

    async def deploy(self, model_id: str, **kwargs: Any) -> bool:
        self.deployed.append(model_id)
        return True

    async def undeploy(self, model_id: str, **kwargs: Any) -> bool:
        self.deployed.remove(model_id)
        return True


class PassValidator:
    def validate(
        self,
        training_result: Any,
        model_card: Any,
        *,
        profile: str,
    ) -> ValidationResult:
        return ValidationResult(valid=True)


class FailValidator:
    def validate(
        self,
        training_result: Any,
        model_card: Any,
        *,
        profile: str,
    ) -> ValidationResult:
        return ValidationResult(valid=False, message="failed gate")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_complete_training() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1",
        weights_hash="abc123",
        metrics={"loss": 0.35},
    )
    assert output.model_id == "m1"
    assert output.weights_hash == "abc123"
    assert output.metrics["loss"] == 0.35
    assert output.correlation_id


def test_complete_training_with_ledger() -> None:
    ledger = FakeLedger()
    coord = GovernanceCoordinator(ledger=ledger)
    coord.complete_training(
        model_id="m1",
        weights_hash="abc",
    )
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["type"] == "training_completed"


def test_submit_for_governance_unsigned() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output, approved_by="alice"
    )
    assert approval.model_id == "m1"
    assert approval.approved_by == "alice"
    assert approval.has_quorum() is True
    assert coord.store.is_approved("m1") is True


def test_submit_with_validator_pass() -> None:
    coord = GovernanceCoordinator(validator=PassValidator())
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output, approved_by="alice"
    )
    assert approval.model_id == "m1"


def test_submit_with_validator_fail() -> None:
    coord = GovernanceCoordinator(validator=FailValidator())
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    with pytest.raises(ValueError, match="validation failed"):
        coord.submit_for_governance(output, approved_by="alice")


def test_submit_with_scope_constraints() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output,
        approved_by="alice",
        approved_environments=["staging"],
        approved_scopes=["scope-a"],
    )
    assert coord.store.is_approved(
        "m1", environment="staging", scope="scope-a"
    )
    assert not coord.store.is_approved(
        "m1", environment="production"
    )


@skip_no_crypto
def test_submit_with_signing() -> None:
    key = Ed25519PrivateKey.generate()
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output,
        approved_by="alice",
        private_key=key,
    )
    assert approval.signature != ""
    assert len(approval.signature) > 0


@skip_no_crypto
def test_multi_approver_quorum() -> None:
    key_a = Ed25519PrivateKey.generate()
    key_b = Ed25519PrivateKey.generate()

    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output,
        approved_by="alice",
        private_key=key_a,
        required_approvers=2,
    )
    assert not coord.store.is_approved("m1")  # Only 1/2

    has_quorum = coord.add_approval_signature(
        "m1", "bob", key_b
    )
    assert has_quorum is True
    assert coord.store.is_approved("m1")


@pytest.mark.asyncio
async def test_full_three_plane_flow() -> None:
    ledger = FakeLedger()
    endpoint = FakeEndpoint()
    coord = GovernanceCoordinator(
        ledger=ledger, endpoint=endpoint
    )

    # Training plane
    output = coord.complete_training(
        model_id="m1",
        weights_hash="abc",
        metrics={"loss": 0.3},
    )

    # Governance plane
    approval = coord.submit_for_governance(
        output, approved_by="alice"
    )

    # Serving plane
    result = await coord.deploy_approved(approval)
    assert result.promoted is True
    assert "m1" in endpoint.deployed

    # Evidence trail
    types = [e["type"] for e in ledger.entries]
    assert "training_completed" in types
    assert "model_approved" in types
    assert "model_deployment" in types


def test_revoke_model() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    coord.submit_for_governance(output, approved_by="alice")
    assert coord.store.is_approved("m1")

    coord.revoke_model("m1", "bob", "security issue")
    assert not coord.store.is_approved("m1")


def test_revoke_with_ledger() -> None:
    ledger = FakeLedger()
    coord = GovernanceCoordinator(ledger=ledger)
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    coord.submit_for_governance(output, approved_by="alice")
    coord.revoke_model("m1", "bob", "reason")

    types = [e["type"] for e in ledger.entries]
    assert "model_approval_revoked" in types


@pytest.mark.asyncio
async def test_deploy_after_revoke_fails() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    approval = coord.submit_for_governance(
        output, approved_by="alice"
    )
    coord.revoke_model("m1", "bob", "reason")

    with pytest.raises(ValueError, match="not approved"):
        await coord.deploy_approved(approval)


def test_drift_no_action() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    coord.submit_for_governance(output, approved_by="alice")

    result = coord.check_drift(
        "m1",
        {"loss": 0.30},
        {"loss": 0.32},
    )
    assert result.is_drifted is False
    assert coord.store.is_approved("m1")


def test_drift_auto_revoke() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    coord.submit_for_governance(output, approved_by="alice")

    result = coord.check_drift(
        "m1",
        {"loss": 0.10},
        {"loss": 1.00},
        auto_revoke=True,
    )
    assert result.is_drifted is True
    assert not coord.store.is_approved("m1")


def test_drift_no_auto_revoke_by_default() -> None:
    coord = GovernanceCoordinator()
    output = coord.complete_training(
        model_id="m1", weights_hash="abc"
    )
    coord.submit_for_governance(output, approved_by="alice")

    result = coord.check_drift(
        "m1",
        {"loss": 0.10},
        {"loss": 1.00},
    )
    assert result.is_drifted is True
    assert coord.store.is_approved("m1")  # NOT revoked
