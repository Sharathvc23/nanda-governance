"""Deployment gate — verify approval before promoting a model.

The ``promote_model()`` function checks expiration, scope, quorum, and
optional signature verification before allowing deployment.  If an
``EvidenceLedger`` or ``ServingEndpoint`` is provided, it records the
event and triggers deployment respectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from nanda_governance.approval import ApprovalStore, ModelApproval
from nanda_governance.contracts import EvidenceLedger, ServingEndpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a model promotion attempt."""

    model_id: str
    promoted: bool
    promoted_at: datetime
    evidence_entry_id: str | None = None
    message: str = ""


async def promote_model(
    approval: ModelApproval,
    store: ApprovalStore,
    *,
    environment: str | None = None,
    scope: str | None = None,
    public_key: Any | None = None,
    ledger: EvidenceLedger | None = None,
    endpoint: ServingEndpoint | None = None,
    correlation_id: str | None = None,
) -> PromotionResult:
    """Promote a model after verifying its governance approval.

    Args:
        approval: The signed approval from the governance plane.
        store: Approval store for status checks.
        environment: Target deployment environment.
        scope: Target deployment scope.
        public_key: Optional Ed25519 public key for signature verification.
        ledger: Optional evidence ledger for audit trail.
        endpoint: Optional serving endpoint for live deployment.
        correlation_id: Correlation ID for evidence logging.

    Returns:
        A ``PromotionResult`` indicating success or failure.

    Raises:
        ValueError: If the approval fails any gate check.
    """
    now = datetime.now(timezone.utc)

    # Gate 1: approval not expired
    if approval.is_expired():
        raise ValueError(
            f"Approval for model {approval.model_id} has expired "
            f"(expired at {approval.expires_at})"
        )

    # Gate 2: valid for environment/scope
    if not approval.is_valid_for(environment, scope):
        raise ValueError(
            f"Approval for model {approval.model_id} not valid for "
            f"environment={environment} scope={scope}"
        )

    # Gate 3: quorum met
    if not approval.has_quorum():
        raise ValueError(
            f"Approval for model {approval.model_id} lacks quorum "
            f"({len(approval.approver_signatures)}/{approval.required_approvers})"
        )

    # Gate 4: store confirms active
    if not store.is_approved(
        approval.model_id, environment=environment, scope=scope
    ):
        raise ValueError(
            f"Model {approval.model_id} not approved in store"
        )

    # Gate 5: optional signature verification
    if public_key is not None:
        from nanda_governance.signing import verify_approval

        if not verify_approval(approval, public_key):
            raise ValueError(
                f"Invalid signature for model {approval.model_id}"
            )

    # Record evidence
    evidence_id: str | None = None
    if ledger is not None:
        evidence_id = ledger.record(
            "model_deployment",
            {
                "model_id": approval.model_id,
                "approved_by": approval.approved_by,
                "environment": environment,
                "scope": scope,
                "correlation_id": correlation_id or approval.correlation_id,
            },
        )

    # Deploy
    if endpoint is not None:
        await endpoint.deploy(approval.model_id)

    logger.info(
        "Promoted model %s (env=%s, scope=%s, correlation_id=%s)",
        approval.model_id,
        environment,
        scope,
        correlation_id or approval.correlation_id,
    )

    return PromotionResult(
        model_id=approval.model_id,
        promoted=True,
        promoted_at=now,
        evidence_entry_id=evidence_id,
        message="Model promoted successfully",
    )


__all__ = [
    "PromotionResult",
    "promote_model",
]
