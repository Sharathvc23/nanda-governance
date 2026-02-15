"""NANDA Model Governance — three-plane ML governance with cryptographic approvals.

Enforces that no single execution path can train, approve, AND deploy
a model.  Three isolated planes — Training, Governance, Serving — with
Ed25519 cryptographic signatures, time-bounded approvals, environment /
scope constraints, M-of-N multi-approver quorum, drift detection, and
revocation.
"""

from __future__ import annotations

from nanda_governance._types import ApprovalStatus, DriftSeverity
from nanda_governance.approval import (
    ApprovalStore,
    ModelApproval,
)
from nanda_governance.contracts import (
    AdapterRegistry,
    EvidenceLedger,
    ModelCard,
    ModelValidator,
    ServingEndpoint,
    TrainingResult,
    ValidationResult,
)
from nanda_governance.coordinator import GovernanceCoordinator
from nanda_governance.drift import (
    DriftAlert,
    DriftCheckResult,
    DriftConfig,
    DriftMetric,
    check_distribution_drift,
    check_drift,
    create_drift_alert,
)
from nanda_governance.promotion import PromotionResult, promote_model
from nanda_governance.signing import sign_approval, verify_approval
from nanda_governance.stores.memory import InMemoryApprovalStore
from nanda_governance.training import TrainingOutput

__version__ = "0.1.0"

__all__ = [
    # coordinator
    "GovernanceCoordinator",
    # enums
    "ApprovalStatus",
    "DriftSeverity",
    # approval
    "ApprovalStore",
    "ModelApproval",
    # training
    "TrainingOutput",
    # signing
    "sign_approval",
    "verify_approval",
    # promotion
    "PromotionResult",
    "promote_model",
    # drift
    "DriftAlert",
    "DriftCheckResult",
    "DriftConfig",
    "DriftMetric",
    "check_distribution_drift",
    "check_drift",
    "create_drift_alert",
    # stores
    "InMemoryApprovalStore",
    # contracts
    "AdapterRegistry",
    "EvidenceLedger",
    "ModelCard",
    "ModelValidator",
    "ServingEndpoint",
    "TrainingResult",
    "ValidationResult",
]
