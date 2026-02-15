"""Tests for ModelApproval creation, serialization, expiration, and scope."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nanda_governance.approval import ModelApproval


def test_approval_defaults() -> None:
    a = ModelApproval(model_id="m1", approved_by="alice")
    assert a.model_id == "m1"
    assert a.approved_by == "alice"
    assert a.status == "active"
    assert a.approval_id.startswith("approval:")
    assert a.required_approvers == 1
    assert a.approved_scopes is None
    assert a.approved_environments is None


def test_compute_hash_deterministic() -> None:
    a = ModelApproval(
        approval_id="fixed",
        model_id="m1",
        weights_hash="abc",
        approved_by="alice",
        approved_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    h1 = a.compute_hash()
    h2 = a.compute_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_hash_excludes_mutable_fields() -> None:
    a = ModelApproval(
        approval_id="fixed",
        model_id="m1",
        approved_by="alice",
        approved_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    h_before = a.compute_hash()
    a.signature = "some-sig"
    a.status = "revoked"
    a.approver_signatures["bob"] = "sig2"
    h_after = a.compute_hash()
    assert h_before == h_after


def test_is_expired_not_expired() -> None:
    a = ModelApproval(
        expires_at=datetime.now(UTC) + timedelta(days=30)
    )
    assert a.is_expired() is False


def test_is_expired_expired() -> None:
    a = ModelApproval(
        expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    assert a.is_expired() is True


def test_is_expired_none() -> None:
    a = ModelApproval(expires_at=None)
    assert a.is_expired() is False


def test_is_valid_for_no_constraints() -> None:
    a = ModelApproval()
    assert a.is_valid_for("production", "scope-x") is True


def test_is_valid_for_environment_match() -> None:
    a = ModelApproval(approved_environments=["staging", "production"])
    assert a.is_valid_for(environment="staging") is True
    assert a.is_valid_for(environment="dev") is False


def test_is_valid_for_scope_match() -> None:
    a = ModelApproval(approved_scopes=["scope-a", "scope-b"])
    assert a.is_valid_for(scope="scope-a") is True
    assert a.is_valid_for(scope="scope-c") is False


def test_is_valid_for_combined() -> None:
    a = ModelApproval(
        approved_environments=["prod"],
        approved_scopes=["s1"],
    )
    assert a.is_valid_for("prod", "s1") is True
    assert a.is_valid_for("prod", "s2") is False
    assert a.is_valid_for("dev", "s1") is False


def test_add_signature_and_quorum() -> None:
    a = ModelApproval(required_approvers=2)
    assert a.has_quorum() is False
    a.add_signature("alice", "sig-a")
    assert a.has_quorum() is False
    a.add_signature("bob", "sig-b")
    assert a.has_quorum() is True


def test_to_dict_from_dict_roundtrip() -> None:
    a = ModelApproval(
        model_id="m1",
        weights_hash="abc",
        approved_by="alice",
        profile="high-risk",
        approved_environments=["staging"],
        approved_scopes=["scope-1"],
        required_approvers=2,
    )
    a.add_signature("alice", "sig-a")
    d = a.to_dict()
    b = ModelApproval.from_dict(d)
    assert b.model_id == a.model_id
    assert b.weights_hash == a.weights_hash
    assert b.approved_by == a.approved_by
    assert b.profile == a.profile
    assert b.approved_environments == a.approved_environments
    assert b.approved_scopes == a.approved_scopes
    assert b.required_approvers == a.required_approvers
    assert b.approver_signatures == a.approver_signatures


def test_from_dict_defaults() -> None:
    a = ModelApproval.from_dict({})
    assert a.model_id == ""
    assert a.profile == "default"
    assert a.status == "active"
    assert a.required_approvers == 1
