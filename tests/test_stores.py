"""Tests for InMemoryApprovalStore CRUD operations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nanda_governance.approval import ModelApproval
from nanda_governance.stores.memory import InMemoryApprovalStore


def test_store_and_get(
    store: InMemoryApprovalStore, sample_approval: ModelApproval
) -> None:
    store.store(sample_approval)
    retrieved = store.get(sample_approval.model_id)
    assert retrieved is not None
    assert retrieved.model_id == sample_approval.model_id
    assert retrieved.approved_by == sample_approval.approved_by


def test_get_nonexistent(store: InMemoryApprovalStore) -> None:
    assert store.get("nonexistent") is None


def test_is_approved_active(
    store: InMemoryApprovalStore, sample_approval: ModelApproval
) -> None:
    store.store(sample_approval)
    assert store.is_approved(sample_approval.model_id) is True


def test_is_approved_revoked(
    store: InMemoryApprovalStore, sample_approval: ModelApproval
) -> None:
    store.store(sample_approval)
    store.revoke(sample_approval.model_id, "bob", "test revocation")
    assert store.is_approved(sample_approval.model_id) is False


def test_is_approved_expired(store: InMemoryApprovalStore) -> None:
    approval = ModelApproval(
        model_id="m-expired",
        approved_by="alice",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    assert store.is_approved("m-expired") is False


def test_is_approved_wrong_environment(
    store: InMemoryApprovalStore,
) -> None:
    approval = ModelApproval(
        model_id="m-env",
        approved_by="alice",
        approved_environments=["staging"],
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    assert store.is_approved("m-env", environment="staging") is True
    assert store.is_approved("m-env", environment="production") is False


def test_is_approved_wrong_scope(
    store: InMemoryApprovalStore,
) -> None:
    approval = ModelApproval(
        model_id="m-scope",
        approved_by="alice",
        approved_scopes=["scope-a"],
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    assert store.is_approved("m-scope", scope="scope-a") is True
    assert store.is_approved("m-scope", scope="scope-b") is False


def test_is_approved_no_quorum(store: InMemoryApprovalStore) -> None:
    approval = ModelApproval(
        model_id="m-quorum",
        approved_by="alice",
        required_approvers=3,
    )
    approval.add_signature("alice", "sig")
    store.store(approval)
    assert store.is_approved("m-quorum") is False


def test_revoke_returns_true(
    store: InMemoryApprovalStore, sample_approval: ModelApproval
) -> None:
    store.store(sample_approval)
    result = store.revoke(
        sample_approval.model_id, "bob", "security issue"
    )
    assert result is True


def test_revoke_nonexistent_returns_false(
    store: InMemoryApprovalStore,
) -> None:
    result = store.revoke("nonexistent", "bob", "reason")
    assert result is False


def test_overwrite_approval(store: InMemoryApprovalStore) -> None:
    a1 = ModelApproval(
        model_id="m1", approved_by="alice", profile="v1"
    )
    a1.add_signature("alice", "sig")
    store.store(a1)

    a2 = ModelApproval(
        model_id="m1", approved_by="bob", profile="v2"
    )
    a2.add_signature("bob", "sig")
    store.store(a2)

    retrieved = store.get("m1")
    assert retrieved is not None
    assert retrieved.approved_by == "bob"
    assert retrieved.profile == "v2"
