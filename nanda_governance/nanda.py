"""Optional bridge to ``nanda-model-integrity-layer``.

Provides helper functions to produce/consume types from the integrity
layer, making it easy to combine governance approvals with provenance
and attestation data.

Requires the ``integrity`` extra::

    pip install nanda-model-governance[integrity]
"""

from __future__ import annotations

from typing import Any

from nanda_governance._compat import has_nanda_integrity
from nanda_governance.approval import ModelApproval


def approval_to_integrity_facts(
    approval: ModelApproval,
) -> dict[str, Any]:
    """Convert a ModelApproval into facts suitable for
    ``nanda_integrity.attach_to_agent_facts()``.

    Returns a dictionary with governance-specific keys that can be
    merged into an agent's fact set.
    """
    return {
        "governance": {
            "approval_id": approval.approval_id,
            "model_id": approval.model_id,
            "weights_hash": approval.weights_hash,
            "approved_by": approval.approved_by,
            "approved_at": approval.approved_at.isoformat(),
            "expires_at": (
                approval.expires_at.isoformat()
                if approval.expires_at
                else None
            ),
            "status": approval.status,
            "profile": approval.profile,
            "has_quorum": approval.has_quorum(),
            "approved_environments": approval.approved_environments,
            "approved_scopes": approval.approved_scopes,
        }
    }


def create_provenance_with_approval(
    approval: ModelApproval,
    **provenance_kwargs: Any,
) -> Any:
    """Create a ``ModelProvenance`` with governance metadata attached.

    Requires the ``nanda-model-integrity-layer`` package.

    Args:
        approval: A governance approval to embed.
        **provenance_kwargs: Arguments forwarded to ``ModelProvenance()``.

    Returns:
        A ``ModelProvenance`` instance.

    Raises:
        ImportError: If ``nanda-model-integrity-layer`` is not installed.
    """
    if not has_nanda_integrity():
        raise ImportError(
            "This function requires the 'nanda-model-integrity-layer' "
            "package. Install it with: "
            "pip install nanda-model-governance[integrity]"
        )

    from nanda_integrity import ModelProvenance  # type: ignore[import-not-found]

    # Merge governance metadata
    extra = provenance_kwargs.pop("extra_metadata", {})
    extra["governance"] = approval_to_integrity_facts(approval)[
        "governance"
    ]

    return ModelProvenance(
        weights_hash=provenance_kwargs.pop(
            "weights_hash", approval.weights_hash
        ),
        extra_metadata=extra,
        **provenance_kwargs,
    )


__all__ = [
    "approval_to_integrity_facts",
    "create_provenance_with_approval",
]
