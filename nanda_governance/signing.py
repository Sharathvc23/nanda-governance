"""Ed25519 signing and verification for model approvals.

Requires the ``cryptography`` extra::

    pip install nanda-model-governance[crypto]

All functions lazily import ``cryptography`` and raise a clear
``ImportError`` if the package is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanda_governance.approval import ModelApproval


def _require_cryptography() -> None:
    from nanda_governance._compat import has_cryptography

    if not has_cryptography():
        raise ImportError(
            "Ed25519 signing requires the 'cryptography' package. "
            "Install it with: pip install nanda-model-governance[crypto]"
        )


def sign_approval(approval: ModelApproval, private_key: Any) -> str:
    """Sign a model approval with Ed25519.

    Args:
        approval: The approval to sign.
        private_key: An ``Ed25519PrivateKey`` instance.

    Returns:
        Hex-encoded signature string.

    Raises:
        ImportError: If ``cryptography`` is not installed.
    """
    _require_cryptography()
    digest = approval.compute_hash()
    sig_bytes: bytes = private_key.sign(digest.encode("utf-8"))
    return sig_bytes.hex()


def verify_approval(approval: ModelApproval, public_key: Any) -> bool:
    """Verify the Ed25519 signature on a model approval.

    Args:
        approval: The approval to verify.
        public_key: An ``Ed25519PublicKey`` instance.

    Returns:
        True if the signature is valid.

    Raises:
        ImportError: If ``cryptography`` is not installed.
    """
    _require_cryptography()
    if not approval.signature:
        return False
    digest = approval.compute_hash()

    from cryptography.exceptions import InvalidSignature

    try:
        public_key.verify(
            bytes.fromhex(approval.signature), digest.encode("utf-8")
        )
        return True
    except (InvalidSignature, ValueError):
        return False


__all__ = [
    "sign_approval",
    "verify_approval",
]
