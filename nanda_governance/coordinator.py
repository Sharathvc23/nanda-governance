"""GovernanceCoordinator — three-plane model lifecycle orchestrator.

Enforces the split architecture:

    Training Plane  ->  Governance Plane  ->  Serving Plane

No single call path can train, approve, AND deploy — each handoff is
a distinct method with its own evidence logging and verification.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from nanda_governance.approval import (
    DEFAULT_APPROVAL_TTL_DAYS,
    ApprovalStore,
    ModelApproval,
)
from nanda_governance.contracts import (
    EvidenceLedger,
    ModelValidator,
    ServingEndpoint,
)
from nanda_governance.drift import (
    DriftCheckResult,
    DriftConfig,
    check_drift,
    create_drift_alert,
)
from nanda_governance.promotion import PromotionResult, promote_model
from nanda_governance.stores.memory import InMemoryApprovalStore
from nanda_governance.training import TrainingOutput

logger = logging.getLogger(__name__)


class GovernanceCoordinator:
    """Orchestrates the three-plane model lifecycle.

    Args:
        store: Approval store backend (defaults to in-memory).
        ledger: Optional evidence ledger for audit trails.
        validator: Optional model validator for governance gates.
        endpoint: Optional serving endpoint for live deployment.
    """

    def __init__(
        self,
        store: ApprovalStore | None = None,
        ledger: EvidenceLedger | None = None,
        validator: ModelValidator | None = None,
        endpoint: ServingEndpoint | None = None,
    ) -> None:
        self._store: ApprovalStore = store or InMemoryApprovalStore()
        self._ledger = ledger
        self._validator = validator
        self._endpoint = endpoint

    @property
    def store(self) -> ApprovalStore:
        """Access the approval store."""
        return self._store

    # -----------------------------------------------------------------
    # Training Plane
    # -----------------------------------------------------------------

    def complete_training(
        self,
        model_id: str,
        weights_hash: str,
        metrics: dict[str, Any] | None = None,
        card: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingOutput:
        """Training Plane exit — produce a handoff for the governance plane.

        Args:
            model_id: Trained model identifier.
            weights_hash: SHA-256 hex digest of model weights.
            metrics: Training metrics (loss, accuracy, etc.).
            card: Model card data as a dictionary.
            correlation_id: Optional correlation ID for tracing.
            metadata: Additional metadata.

        Returns:
            A ``TrainingOutput`` for submission to the governance plane.
        """
        from uuid import uuid4

        output = TrainingOutput(
            model_id=model_id,
            weights_hash=weights_hash,
            metrics=metrics or {},
            card=card or {},
            correlation_id=correlation_id or uuid4().hex[:16],
            metadata=metadata or {},
        )

        if self._ledger is not None:
            self._ledger.record(
                "training_completed",
                {
                    "model_id": model_id,
                    "weights_hash": weights_hash,
                    "metrics": metrics or {},
                    "correlation_id": output.correlation_id,
                },
            )

        logger.info(
            "Training plane complete: model=%s correlation_id=%s",
            model_id,
            output.correlation_id,
        )

        return output

    # -----------------------------------------------------------------
    # Governance Plane
    # -----------------------------------------------------------------

    def submit_for_governance(
        self,
        training_output: TrainingOutput,
        *,
        approved_by: str,
        profile: str = "default",
        private_key: Any | None = None,
        approval_ttl_days: int | None = DEFAULT_APPROVAL_TTL_DAYS,
        approved_environments: list[str] | None = None,
        approved_scopes: list[str] | None = None,
        required_approvers: int = 1,
    ) -> ModelApproval:
        """Governance Plane — validate, create approval, optionally sign.

        Args:
            training_output: Handoff from the training plane.
            approved_by: Identifier of the governance approver.
            profile: Governance profile for gate validation.
            private_key: Optional Ed25519 private key for signing.
            approval_ttl_days: Days until expiration (None = never).
            approved_environments: Restrict to these environments.
            approved_scopes: Restrict to these scopes.
            required_approvers: Signatures required for quorum.

        Returns:
            A ``ModelApproval`` (signed if private_key provided).

        Raises:
            ValueError: If the model fails governance validation.
        """
        # Optional validation gate
        if self._validator is not None:
            # Build minimal protocol-compatible objects

            result = self._validator.validate(
                _SimpleTrainingResult(
                    training_output.metrics,
                    training_output.model_id,
                ),
                _SimpleModelCard(
                    training_output.model_id,
                    training_output.weights_hash,
                    training_output.card,
                ),
                profile=profile,
            )
            if not result.valid:
                raise ValueError(
                    f"Governance validation failed for "
                    f"{training_output.model_id}: {result.message}"
                )

        expires_at = None
        if approval_ttl_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=approval_ttl_days
            )

        approval = ModelApproval(
            model_id=training_output.model_id,
            weights_hash=training_output.weights_hash,
            approved_by=approved_by,
            expires_at=expires_at,
            profile=profile,
            correlation_id=training_output.correlation_id,
            approved_environments=approved_environments,
            approved_scopes=approved_scopes,
            required_approvers=required_approvers,
        )

        # Sign if key provided
        if private_key is not None:
            from nanda_governance.signing import sign_approval

            approval.signature = sign_approval(approval, private_key)

        # First approver's signature
        approval.add_signature(
            approved_by, approval.signature or "unsigned"
        )

        self._store.store(approval)

        if self._ledger is not None:
            self._ledger.record(
                "model_approved",
                {
                    "model_id": approval.model_id,
                    "approved_by": approved_by,
                    "profile": profile,
                    "expires_at": (
                        expires_at.isoformat() if expires_at else None
                    ),
                    "correlation_id": training_output.correlation_id,
                },
            )

        logger.info(
            "Governance approved model %s (by %s, profile=%s, expires=%s)",
            approval.model_id,
            approved_by,
            profile,
            expires_at.isoformat() if expires_at else "never",
        )

        return approval

    def add_approval_signature(
        self,
        model_id: str,
        approver_id: str,
        private_key: Any,
    ) -> bool:
        """Add an additional signature for multi-approver quorum.

        Args:
            model_id: The model to add a signature for.
            approver_id: Identifier of the additional approver.
            private_key: Ed25519 private key for signing.

        Returns:
            True if quorum is now met, False otherwise.

        Raises:
            ValueError: If no approval exists for the model.
        """
        approval = self._store.get(model_id)
        if approval is None:
            raise ValueError(f"No approval found for model {model_id}")

        from nanda_governance.signing import sign_approval

        signature = sign_approval(approval, private_key)
        approval.add_signature(approver_id, signature)

        self._store.store(approval)

        has_quorum = approval.has_quorum()
        logger.info(
            "Added signature for model %s (by %s, quorum=%s, %d/%d)",
            model_id,
            approver_id,
            has_quorum,
            len(approval.approver_signatures),
            approval.required_approvers,
        )

        return has_quorum

    # -----------------------------------------------------------------
    # Serving Plane
    # -----------------------------------------------------------------

    async def deploy_approved(
        self,
        approval: ModelApproval,
        *,
        environment: str | None = None,
        scope: str | None = None,
        public_key: Any | None = None,
    ) -> PromotionResult:
        """Serving Plane entry — verify approval and deploy.

        Args:
            approval: Signed approval from the governance plane.
            environment: Target deployment environment.
            scope: Target deployment scope.
            public_key: Optional Ed25519 public key for verification.

        Returns:
            A ``PromotionResult`` from the promotion.

        Raises:
            ValueError: If any gate check fails.
        """
        return await promote_model(
            approval,
            self._store,
            environment=environment,
            scope=scope,
            public_key=public_key,
            ledger=self._ledger,
            endpoint=self._endpoint,
            correlation_id=approval.correlation_id,
        )

    # -----------------------------------------------------------------
    # Revocation
    # -----------------------------------------------------------------

    def revoke_model(
        self,
        model_id: str,
        revoked_by: str,
        reason: str,
    ) -> None:
        """Revoke a model's governance approval.

        Args:
            model_id: The model to revoke.
            revoked_by: Identifier of the revoking entity.
            reason: Human-readable revocation reason.
        """
        self._store.revoke(model_id, revoked_by, reason)

        if self._ledger is not None:
            approval = self._store.get(model_id)
            self._ledger.record(
                "model_approval_revoked",
                {
                    "model_id": model_id,
                    "revoked_by": revoked_by,
                    "reason": reason,
                    "correlation_id": (
                        approval.correlation_id if approval else None
                    ),
                },
            )

        logger.info(
            "Revoked model %s (by %s, reason: %s)",
            model_id,
            revoked_by,
            reason,
        )

    # -----------------------------------------------------------------
    # Drift Detection
    # -----------------------------------------------------------------

    def check_drift(
        self,
        model_id: str,
        training_metrics: dict[str, float],
        serving_metrics: dict[str, float],
        *,
        auto_revoke: bool = False,
        config: DriftConfig | None = None,
    ) -> DriftCheckResult:
        """Check for model drift and optionally auto-revoke.

        Args:
            model_id: The model to check.
            training_metrics: Baseline metrics from training.
            serving_metrics: Current metrics from production.
            auto_revoke: Auto-revoke on severe drift (default False).
            config: Optional drift configuration.

        Returns:
            DriftCheckResult with drift assessment.
        """
        result = check_drift(
            model_id, training_metrics, serving_metrics, config=config
        )

        if self._ledger is not None and result.is_drifted:
            approval = self._store.get(model_id)
            self._ledger.record(
                "model_drift_detected",
                {
                    "model_id": model_id,
                    "severity": result.overall_severity,
                    "confidence": result.confidence,
                    "summary": result.summary,
                    "recommended_action": result.recommended_action,
                    "correlation_id": (
                        approval.correlation_id if approval else None
                    ),
                },
            )

            alert = create_drift_alert(result)
            if alert:
                logger.warning(
                    "Model drift detected: %s (severity=%s, action=%s)",
                    model_id,
                    alert.severity,
                    result.recommended_action,
                )

        if (
            auto_revoke
            and result.is_drifted
            and result.overall_severity >= 0.8
        ):
            logger.warning(
                "Auto-revoking model %s due to severe drift "
                "(severity=%.2f)",
                model_id,
                result.overall_severity,
            )
            self.revoke_model(
                model_id,
                revoked_by="system:drift-detector",
                reason=(
                    f"Automatic revocation due to drift: {result.summary}"
                ),
            )

        return result


# ---------------------------------------------------------------------------
# Internal helpers for protocol adaptation
# ---------------------------------------------------------------------------


class _SimpleTrainingResult:
    """Minimal TrainingResult for validator protocol."""

    def __init__(
        self, metrics: dict[str, Any], model_id: str
    ) -> None:
        self._metrics = metrics
        self._model_id = model_id

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    @property
    def model_id(self) -> str:
        return self._model_id


class _SimpleModelCard:
    """Minimal ModelCard for validator protocol."""

    def __init__(
        self,
        model_id: str,
        weights_hash: str,
        card: dict[str, Any],
    ) -> None:
        self._model_id = model_id
        self._weights_hash = weights_hash
        self._card = card

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "weights_hash": self._weights_hash,
            **self._card,
        }


__all__ = ["GovernanceCoordinator"]
