"""Shared enumerations for governance and drift detection."""

from __future__ import annotations

from enum import Enum


class ApprovalStatus(str, Enum):
    """Status of a model approval."""

    ACTIVE = "active"
    REVOKED = "revoked"


class DriftSeverity(str, Enum):
    """Severity level for drift alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
