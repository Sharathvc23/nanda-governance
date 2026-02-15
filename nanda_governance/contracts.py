"""Protocol definitions for pluggable backends.

Every external dependency is expressed as a ``@runtime_checkable Protocol``
so that users can bring their own model registry, evidence ledger, serving
endpoint, or validator without inheriting from a base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelCard(Protocol):
    """Minimal model-card interface required by the governance system."""

    @property
    def model_id(self) -> str: ...

    @property
    def weights_hash(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Training Result
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingResult(Protocol):
    """Minimal training-result interface for governance validation."""

    @property
    def metrics(self) -> dict[str, Any]: ...

    @property
    def model_id(self) -> str: ...


# ---------------------------------------------------------------------------
# Adapter Registry
# ---------------------------------------------------------------------------


@runtime_checkable
class AdapterRegistry(Protocol):
    """Backend for registering and retrieving model adapters."""

    async def register(self, metadata: Any) -> Any: ...

    async def get(self, adapter_id: str, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Evidence Ledger
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceLedger(Protocol):
    """Append-only audit log.

    Entry types are plain strings (e.g., ``"training_completed"``,
    ``"model_deployment"``) rather than enum members, keeping the
    ledger decoupled from any specific enum definition.
    """

    def record(self, entry_type: str, data: dict[str, Any]) -> str:
        """Record an event and return its entry ID."""
        ...


# ---------------------------------------------------------------------------
# Serving Endpoint
# ---------------------------------------------------------------------------


@runtime_checkable
class ServingEndpoint(Protocol):
    """Interface for deploying / undeploying a model."""

    async def deploy(self, model_id: str, **kwargs: Any) -> bool: ...

    async def undeploy(self, model_id: str, **kwargs: Any) -> bool: ...


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelValidator(Protocol):
    """Pluggable model validator (e.g., gate checks before approval)."""

    def validate(
        self,
        training_result: TrainingResult,
        model_card: ModelCard,
        *,
        profile: str,
    ) -> ValidationResult: ...


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a model validation check."""

    valid: bool
    message: str = ""


__all__ = [
    "AdapterRegistry",
    "EvidenceLedger",
    "ModelCard",
    "ModelValidator",
    "ServingEndpoint",
    "TrainingResult",
    "ValidationResult",
]
