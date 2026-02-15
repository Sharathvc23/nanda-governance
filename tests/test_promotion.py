"""Tests for the promotion gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nanda_governance.approval import ModelApproval
from nanda_governance.promotion import promote_model
from nanda_governance.stores.memory import InMemoryApprovalStore


@pytest.fixture
def active_store() -> tuple[InMemoryApprovalStore, ModelApproval]:
    """Store with one valid approval pre-loaded."""
    store = InMemoryApprovalStore()
    approval = ModelApproval(
        model_id="m1",
        weights_hash="abc",
        approved_by="alice",
        approved_environments=["staging", "production"],
        approved_scopes=["scope-1"],
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    return store, approval


@pytest.mark.asyncio
async def test_promote_valid(
    active_store: tuple[InMemoryApprovalStore, ModelApproval],
) -> None:
    store, approval = active_store
    result = await promote_model(
        approval,
        store,
        environment="staging",
        scope="scope-1",
    )
    assert result.promoted is True
    assert result.model_id == "m1"


@pytest.mark.asyncio
async def test_promote_expired() -> None:
    store = InMemoryApprovalStore()
    approval = ModelApproval(
        model_id="m1",
        approved_by="alice",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    approval.add_signature("alice", "sig")
    store.store(approval)

    with pytest.raises(ValueError, match="expired"):
        await promote_model(approval, store)


@pytest.mark.asyncio
async def test_promote_wrong_scope(
    active_store: tuple[InMemoryApprovalStore, ModelApproval],
) -> None:
    store, approval = active_store
    with pytest.raises(ValueError, match="not valid"):
        await promote_model(
            approval, store, scope="wrong-scope"
        )


@pytest.mark.asyncio
async def test_promote_wrong_environment(
    active_store: tuple[InMemoryApprovalStore, ModelApproval],
) -> None:
    store, approval = active_store
    with pytest.raises(ValueError, match="not valid"):
        await promote_model(
            approval, store, environment="dev"
        )


@pytest.mark.asyncio
async def test_promote_revoked() -> None:
    store = InMemoryApprovalStore()
    approval = ModelApproval(
        model_id="m1",
        approved_by="alice",
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    store.revoke("m1", "bob", "security")

    with pytest.raises(ValueError, match="not approved"):
        await promote_model(approval, store)


@pytest.mark.asyncio
async def test_promote_no_quorum() -> None:
    store = InMemoryApprovalStore()
    approval = ModelApproval(
        model_id="m1",
        approved_by="alice",
        required_approvers=3,
    )
    approval.add_signature("alice", "sig-a")
    store.store(approval)

    with pytest.raises(ValueError, match="quorum"):
        await promote_model(approval, store)


@pytest.mark.asyncio
async def test_promote_with_ledger(
    active_store: tuple[InMemoryApprovalStore, ModelApproval],
) -> None:
    store, approval = active_store

    class FakeLedger:
        def __init__(self) -> None:
            self.entries: list[dict[str, object]] = []

        def record(
            self, entry_type: str, data: dict[str, object]
        ) -> str:
            self.entries.append(
                {"type": entry_type, "data": data}
            )
            return f"entry-{len(self.entries)}"

    ledger = FakeLedger()
    result = await promote_model(
        approval,
        store,
        environment="staging",
        scope="scope-1",
        ledger=ledger,
    )
    assert result.evidence_entry_id == "entry-1"
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["type"] == "model_deployment"
