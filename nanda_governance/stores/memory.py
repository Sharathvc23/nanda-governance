"""In-memory approval store — zero external dependencies.

Thread-safe via a simple lock. Suitable for testing, local development,
and single-process deployments.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from nanda_governance.approval import ModelApproval

logger = logging.getLogger(__name__)


class InMemoryApprovalStore:
    """In-memory store for model approvals, keyed by ``model_id``.

    Thread-safe: all mutations are guarded by an internal lock.
    """

    def __init__(self) -> None:
        self._approvals: dict[str, ModelApproval] = {}
        self._lock = threading.Lock()

    def store(self, approval: ModelApproval) -> None:
        """Store an approval, keyed by ``model_id``."""
        with self._lock:
            self._approvals[approval.model_id] = approval
        logger.info(
            "Stored approval %s for model %s (by %s)",
            approval.approval_id,
            approval.model_id,
            approval.approved_by,
        )

    def get(self, model_id: str) -> ModelApproval | None:
        """Retrieve the approval for a model."""
        with self._lock:
            return self._approvals.get(model_id)

    def is_approved(
        self,
        model_id: str,
        *,
        environment: str | None = None,
        scope: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        """Check whether a model has an active, unexpired approval.

        Validates status, expiration, scope constraints, and quorum.
        """
        with self._lock:
            approval = self._approvals.get(model_id)
        if approval is None:
            return False
        if approval.status != "active":
            return False
        if approval.is_expired():
            logger.warning(
                "Approval for model %s has expired (expires_at=%s)",
                model_id,
                approval.expires_at,
            )
            return False
        if not approval.is_valid_for(environment, scope):
            logger.warning(
                "Approval for model %s not valid for "
                "environment=%s scope=%s",
                model_id,
                environment,
                scope,
            )
            return False
        if not approval.has_quorum():
            logger.warning(
                "Approval for model %s lacks quorum (%d/%d signatures)",
                model_id,
                len(approval.approver_signatures),
                approval.required_approvers,
            )
            return False
        return True

    def revoke(
        self, model_id: str, revoked_by: str, reason: str
    ) -> bool:
        """Revoke an existing approval."""
        with self._lock:
            approval = self._approvals.get(model_id)
            if approval is None:
                return False
            approval.status = "revoked"
        logger.info(
            "Revoked approval for model %s (by %s, reason: %s)",
            model_id,
            revoked_by,
            reason,
        )
        return True


__all__ = ["InMemoryApprovalStore"]
