"""Training Plane exit — the handoff object to the Governance Plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class TrainingOutput:
    """Handoff object from the Training Plane to the Governance Plane.

    Bundles the model identity, weights hash, card data, metrics, and
    correlation ID so that the governance plane has everything it needs
    to validate and approve the model.
    """

    model_id: str
    weights_hash: str
    card: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: uuid4().hex[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "model_id": self.model_id,
            "weights_hash": self.weights_hash,
            "card": self.card,
            "metrics": self.metrics,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingOutput:
        """Deserialize from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(timezone.utc)

        return cls(
            model_id=data.get("model_id", ""),
            weights_hash=data.get("weights_hash", ""),
            card=data.get("card", {}),
            metrics=data.get("metrics", {}),
            correlation_id=data.get("correlation_id", uuid4().hex[:16]),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


__all__ = ["TrainingOutput"]
