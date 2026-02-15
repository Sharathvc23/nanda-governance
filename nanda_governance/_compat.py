"""Optional dependency detection.

Centralises availability checks so that each module can import
``has_cryptography()`` etc. instead of scattering try/except blocks.
"""

from __future__ import annotations


def has_cryptography() -> bool:
    """Return True if the ``cryptography`` package is importable."""
    try:
        import cryptography  # noqa: F401

        return True
    except ImportError:
        return False


def has_psycopg2() -> bool:
    """Return True if ``psycopg2`` is importable."""
    try:
        import psycopg2  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def has_nanda_integrity() -> bool:
    """Return True if ``nanda-model-integrity-layer`` is importable."""
    try:
        import nanda_integrity  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False
