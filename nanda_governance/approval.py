"""Governance gate core — ModelApproval dataclass and ApprovalStore protocol.

Each approval is a signed record tying a model identity + weights hash
to an approver, with time-bounded expiration, environment/scope constraints,
and M-of-N multi-approver quorum support.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_APPROVAL_TTL_DAYS = 90


@dataclass
class ModelApproval:
    """A cryptographically signed model approval record.

    Attributes:
        approval_id: Unique approval identifier.
        model_id: The model this approval covers.
        weights_hash: SHA-256 hex digest of the model weights.
        approved_by: Identifier of the governance approver.
        approved_at: Timestamp of approval.
        expires_at: Approval expiration (None = never expires).
        profile: Governance profile under which approval was granted.
        correlation_id: Correlation ID threading from training.
        signature: Hex-encoded Ed25519 signature over the approval hash.
        status: ``"active"`` or ``"revoked"``.
        approved_environments: Environments where valid (None = all).
        approved_scopes: Scopes where valid (None = all).
        required_approvers: Signatures required for quorum (default 1).
        approver_signatures: Mapping of approver ID -> hex signature.
    """

    approval_id: str = field(
        default_factory=lambda: f"approval:{uuid4().hex[:16]}"
    )
    model_id: str = ""
    weights_hash: str = ""
    approved_by: str = ""
    approved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime | None = None
    profile: str = "default"
    correlation_id: str | None = None
    signature: str = ""
    status: str = "active"
    approved_environments: list[str] | None = None
    approved_scopes: list[str] | None = None
    required_approvers: int = 1
    approver_signatures: dict[str, str] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute SHA-256 of canonical approval content.

        Covers all fields except ``signature``, ``status``, and
        ``approver_signatures`` so signing is deterministic.
        """
        canonical = json.dumps(
            {
                "approval_id": self.approval_id,
                "model_id": self.model_id,
                "weights_hash": self.weights_hash,
                "approved_by": self.approved_by,
                "approved_at": self.approved_at.isoformat(),
                "expires_at": (
                    self.expires_at.isoformat() if self.expires_at else None
                ),
                "profile": self.profile,
                "correlation_id": self.correlation_id,
                "approved_environments": self.approved_environments,
                "approved_scopes": self.approved_scopes,
                "required_approvers": self.required_approvers,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_expired(self) -> bool:
        """Return True if this approval has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid_for(
        self,
        environment: str | None = None,
        scope: str | None = None,
    ) -> bool:
        """Check if this approval covers the given environment and scope."""
        if (
            environment is not None
            and self.approved_environments is not None
            and environment not in self.approved_environments
        ):
            return False
        if (
            scope is not None
            and self.approved_scopes is not None
            and scope not in self.approved_scopes
        ):
            return False
        return True

    def add_signature(self, approver_id: str, signature: str) -> None:
        """Add an approver's signature for multi-sig governance."""
        self.approver_signatures[approver_id] = signature

    def has_quorum(self) -> bool:
        """Return True if enough signatures have been collected."""
        return len(self.approver_signatures) >= self.required_approvers

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "approval_id": self.approval_id,
            "model_id": self.model_id,
            "weights_hash": self.weights_hash,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
            "profile": self.profile,
            "correlation_id": self.correlation_id,
            "signature": self.signature,
            "status": self.status,
            "approved_environments": self.approved_environments,
            "approved_scopes": self.approved_scopes,
            "required_approvers": self.required_approvers,
            "approver_signatures": self.approver_signatures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelApproval:
        """Deserialize from a dictionary."""
        approved_at = data.get("approved_at")
        if isinstance(approved_at, str):
            approved_at = datetime.fromisoformat(approved_at)
        else:
            approved_at = datetime.now(timezone.utc)

        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        else:
            expires_at = None

        return cls(
            approval_id=data.get(
                "approval_id", f"approval:{uuid4().hex[:16]}"
            ),
            model_id=data.get("model_id", ""),
            weights_hash=data.get("weights_hash", ""),
            approved_by=data.get("approved_by", ""),
            approved_at=approved_at,
            expires_at=expires_at,
            profile=data.get("profile", "default"),
            correlation_id=data.get("correlation_id"),
            signature=data.get("signature", ""),
            status=data.get("status", "active"),
            approved_environments=data.get("approved_environments"),
            approved_scopes=data.get("approved_scopes"),
            required_approvers=data.get("required_approvers", 1),
            approver_signatures=data.get("approver_signatures", {}),
        )


# ---------------------------------------------------------------------------
# ApprovalStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalStore(Protocol):
    """Protocol for approval persistence backends."""

    def store(self, approval: ModelApproval) -> None: ...

    def get(self, model_id: str) -> ModelApproval | None: ...

    def is_approved(
        self,
        model_id: str,
        *,
        environment: str | None = None,
        scope: str | None = None,
    ) -> bool: ...

    def revoke(
        self, model_id: str, revoked_by: str, reason: str
    ) -> bool: ...


__all__ = [
    "ApprovalStore",
    "DEFAULT_APPROVAL_TTL_DAYS",
    "ModelApproval",
]
