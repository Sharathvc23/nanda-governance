"""Tests for Ed25519 signing and verification."""

from __future__ import annotations

import pytest

from nanda_governance.approval import ModelApproval
from tests.conftest import HAS_CRYPTOGRAPHY, skip_no_crypto

if HAS_CRYPTOGRAPHY:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )


@skip_no_crypto
def test_sign_verify_roundtrip() -> None:
    from nanda_governance.signing import sign_approval, verify_approval

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    approval = ModelApproval(
        model_id="m1",
        weights_hash="abc",
        approved_by="alice",
    )
    sig = sign_approval(approval, private_key)
    assert isinstance(sig, str)
    assert len(sig) > 0

    approval.signature = sig
    assert verify_approval(approval, public_key) is True


@skip_no_crypto
def test_tamper_detection() -> None:
    from nanda_governance.signing import sign_approval, verify_approval

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    approval = ModelApproval(
        model_id="m1",
        weights_hash="abc",
        approved_by="alice",
    )
    approval.signature = sign_approval(approval, private_key)

    # Tamper with model_id
    approval.model_id = "tampered"
    assert verify_approval(approval, public_key) is False


@skip_no_crypto
def test_wrong_key_fails() -> None:
    from nanda_governance.signing import sign_approval, verify_approval

    key_a = Ed25519PrivateKey.generate()
    key_b = Ed25519PrivateKey.generate()

    approval = ModelApproval(
        model_id="m1",
        approved_by="alice",
    )
    approval.signature = sign_approval(approval, key_a)

    # Verify with wrong public key
    assert verify_approval(approval, key_b.public_key()) is False


@skip_no_crypto
def test_empty_signature_returns_false() -> None:
    from nanda_governance.signing import verify_approval

    key = Ed25519PrivateKey.generate()
    approval = ModelApproval(model_id="m1", signature="")
    assert verify_approval(approval, key.public_key()) is False


def test_import_error_without_crypto() -> None:
    """Verify that helpful ImportError is raised when crypto is missing."""
    from nanda_governance._compat import has_cryptography

    if has_cryptography():
        pytest.skip("cryptography is installed")

    from nanda_governance.signing import sign_approval

    approval = ModelApproval(model_id="m1")
    with pytest.raises(ImportError, match="cryptography"):
        sign_approval(approval, None)
