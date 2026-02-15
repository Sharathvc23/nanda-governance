"""PostgreSQL-backed approval store for production deployments.

Requires the ``postgres`` extra::

    pip install nanda-model-governance[postgres]

Uses ``psycopg2``'s ``ThreadedConnectionPool`` for thread-safe
concurrent access.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from nanda_governance._compat import has_psycopg2
from nanda_governance.approval import ModelApproval

logger = logging.getLogger(__name__)


APPROVAL_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_approvals (
    approval_id VARCHAR(64) PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL,
    weights_hash VARCHAR(64),
    approved_by VARCHAR(256) NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ,
    profile VARCHAR(64) DEFAULT 'default',
    correlation_id VARCHAR(64),
    signature TEXT NOT NULL,
    status VARCHAR(16) DEFAULT 'active',
    approved_environments TEXT[],
    approved_scopes TEXT[],
    required_approvers INTEGER DEFAULT 1,
    approver_signatures JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_approvals_model_id
    ON model_approvals(model_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status
    ON model_approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_expires
    ON model_approvals(expires_at);
"""


class PostgresApprovalStore:
    """PostgreSQL-backed storage for model approvals.

    Thread-safe via ``psycopg2.pool.ThreadedConnectionPool``.

    Args:
        connection_string: PostgreSQL DSN / connection URL.
        pool_min: Minimum connections in the pool.
        pool_max: Maximum connections in the pool.
    """

    def __init__(
        self,
        *,
        connection_string: str,
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        if not has_psycopg2():
            raise ImportError(
                "PostgresApprovalStore requires the 'psycopg2' package. "
                "Install it with: "
                "pip install nanda-model-governance[postgres]"
            )

        import psycopg2.pool  # type: ignore[import-untyped]

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            pool_min, pool_max, connection_string
        )
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._transaction() as cur:
            for statement in APPROVAL_PG_SCHEMA.split(";"):
                statement = statement.strip()
                if statement:
                    cur.execute(statement)

    @contextmanager
    def _transaction(self) -> Generator[Any, None, None]:
        """Check out a connection, yield a cursor inside a transaction."""
        conn = self._pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                yield cur
        finally:
            self._pool.putconn(conn)

    def _dictfetchone(self, cur: Any) -> dict[str, Any] | None:
        """Fetch one row as a dict."""
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        return dict(zip(cols, row, strict=False))

    # -----------------------------------------------------------------
    # ApprovalStore interface
    # -----------------------------------------------------------------

    def store(self, approval: ModelApproval) -> None:
        """Store (upsert) an approval."""
        from psycopg2.extras import Json  # type: ignore[import-untyped]

        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO model_approvals (
                    approval_id, model_id, weights_hash, approved_by,
                    approved_at, expires_at, profile, correlation_id,
                    signature, status, approved_environments,
                    approved_scopes, required_approvers,
                    approver_signatures, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (approval_id) DO UPDATE SET
                    model_id = EXCLUDED.model_id,
                    weights_hash = EXCLUDED.weights_hash,
                    approved_by = EXCLUDED.approved_by,
                    approved_at = EXCLUDED.approved_at,
                    expires_at = EXCLUDED.expires_at,
                    profile = EXCLUDED.profile,
                    correlation_id = EXCLUDED.correlation_id,
                    signature = EXCLUDED.signature,
                    status = EXCLUDED.status,
                    approved_environments = EXCLUDED.approved_environments,
                    approved_scopes = EXCLUDED.approved_scopes,
                    required_approvers = EXCLUDED.required_approvers,
                    approver_signatures = EXCLUDED.approver_signatures,
                    updated_at = NOW()
                """,
                (
                    approval.approval_id,
                    approval.model_id,
                    approval.weights_hash,
                    approval.approved_by,
                    approval.approved_at,
                    approval.expires_at,
                    approval.profile,
                    approval.correlation_id,
                    approval.signature,
                    approval.status,
                    approval.approved_environments,
                    approval.approved_scopes,
                    approval.required_approvers,
                    Json(approval.approver_signatures),
                ),
            )

        logger.info(
            "Stored approval %s for model %s",
            approval.approval_id,
            approval.model_id,
        )

    def get(self, model_id: str) -> ModelApproval | None:
        """Retrieve the most recent approval for a model."""
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM model_approvals
                WHERE model_id = %s
                ORDER BY approved_at DESC
                LIMIT 1
                """,
                (model_id,),
            )
            row = self._dictfetchone(cur)

        if row is None:
            return None
        return self._row_to_approval(row)

    def is_approved(
        self,
        model_id: str,
        *,
        environment: str | None = None,
        scope: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        """Check whether a model has an active, valid approval."""
        approval = self.get(model_id)
        if approval is None:
            return False
        if approval.status != "active":
            return False
        if approval.is_expired():
            return False
        if not approval.is_valid_for(environment, scope):
            return False
        if not approval.has_quorum():
            return False
        return True

    def revoke(
        self, model_id: str, revoked_by: str, reason: str
    ) -> bool:
        """Revoke an existing approval."""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE model_approvals
                SET status = 'revoked', updated_at = NOW()
                WHERE model_id = %s AND status = 'active'
                """,
                (model_id,),
            )
            updated: bool = cur.rowcount > 0

        if updated:
            logger.info(
                "Revoked approval for model %s (by %s, reason: %s)",
                model_id,
                revoked_by,
                reason,
            )
        return updated

    def list_expiring(
        self, within_days: int = 7
    ) -> list[ModelApproval]:
        """List approvals expiring within the specified window."""
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM model_approvals
                WHERE status = 'active'
                  AND expires_at IS NOT NULL
                  AND expires_at <= NOW() + INTERVAL '%s days'
                  AND expires_at > NOW()
                ORDER BY expires_at ASC
                """,
                (within_days,),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in cur.description]
            return [
                self._row_to_approval(dict(zip(cols, row, strict=False)))
                for row in rows
            ]

    def _row_to_approval(self, row: dict[str, Any]) -> ModelApproval:
        """Convert a database row to a ModelApproval."""
        approved_at = row["approved_at"]
        if isinstance(approved_at, str):
            approved_at = datetime.fromisoformat(approved_at)

        expires_at = row.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        approver_signatures = row.get("approver_signatures", {})
        if isinstance(approver_signatures, str):
            approver_signatures = json.loads(approver_signatures)

        return ModelApproval(
            approval_id=row["approval_id"],
            model_id=row["model_id"],
            weights_hash=row.get("weights_hash", ""),
            approved_by=row["approved_by"],
            approved_at=approved_at,
            expires_at=expires_at,
            profile=row.get("profile", "default"),
            correlation_id=row.get("correlation_id"),
            signature=row.get("signature", ""),
            status=row.get("status", "active"),
            approved_environments=row.get("approved_environments"),
            approved_scopes=row.get("approved_scopes"),
            required_approvers=row.get("required_approvers", 1),
            approver_signatures=approver_signatures,
        )

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None


__all__ = [
    "APPROVAL_PG_SCHEMA",
    "PostgresApprovalStore",
]
