# Three-Plane Separation for Cryptographic ML Model Governance

**Authors:** StellarMinds ([stellarminds.ai](https://stellarminds.ai))
**Date:** April 2026
**Version:** 0.2.0

## Abstract

`sm-model-governance` is a Python library that enforces three-plane separation (training, governance, serving) for ML model deployment through type-enforced handoffs and cryptographic approval gates. It solves the problem of monolithic ML pipelines where a single actor or code path can unilaterally train, approve, and deploy a model. The library provides Ed25519 signing, a 5-gate promotion check, M-of-N quorum, drift detection, and time-bounded approvals with auto-revocation. The core requires no runtime dependencies, with cryptographic operations available via lazy optional imports.

## Problem

Conventional ML deployment pipelines often allow a single CI/CD job or notebook to train, evaluate, and push a model to production. This conflation of responsibilities creates insider threat risk (compromised actors can substitute weights without review), compliance violations (SOX and NIST SP 800-57 require separation of duties), and stale deployment risk when approvals have no expiration and models drift undetected.

## What It Does

- Enforces 3-plane separation via typed handoffs: `TrainingOutput` -> `ModelApproval` -> `PromotionResult`
- Signs approvals with Ed25519 digital signatures using lazy imports so the core operates with zero dependencies
- Validates deployment through a 5-gate sequential promotion check: expiration, environment/scope, quorum, store confirmation, cryptographic signature
- Supports M-of-N multi-approver quorum where each approver contributes an independent Ed25519 signature over a deterministic canonical hash
- Detects distribution drift via a custom two-sample Kolmogorov-Smirnov statistic using a two-pointer algorithm (no scipy required)
- Issues time-bounded approvals with configurable TTL (default 90 days) and auto-revokes on severe drift (severity >= 0.8)
- Provides a pluggable `ApprovalStore` protocol with thread-safe in-memory and PostgreSQL backends
- Records all governance events to an `EvidenceLedger` protocol for audit tracing across all three planes

## Architecture

The architecture partitions the model lifecycle into three isolated planes, each with a distinct API, output type, and audit event:

```
 Training Plane          Governance Plane           Serving Plane
      |                       |                         |
 complete_training()    submit_for_governance()    deploy_approved()
      |                       |                         |
      v                       v                         v
 TrainingOutput -------> ModelApproval ----------> PromotionResult
                          (signed,                  (5 gates checked:
                           scoped,                   expiry, scope,
                           time-bounded,             quorum, store,
                           M-of-N quorum)            signature)
```

The `GovernanceCoordinator` orchestrates all three planes while maintaining isolation. No single function in the public API spans all three planes. The separation is encouraged through type-level constraints: `deploy_approved()` requires a `ModelApproval` parameter, which is normally obtained from `submit_for_governance()`, which in turn requires a `TrainingOutput` from `complete_training()`. No single function in the public API spans all three planes.

The coordinator accepts four protocol-typed dependencies:

| Dependency | Protocol | Purpose |
|------------|----------|---------|
| `store` | `ApprovalStore` | Approval persistence (default: `InMemoryApprovalStore`) |
| `ledger` | `EvidenceLedger` | Audit trail recording |
| `validator` | `ModelValidator` | Optional pre-governance validation |
| `endpoint` | `ServingEndpoint` | Optional deployment execution |

The 5-gate promotion check validates sequentially before allowing deployment:

| Gate | Check | Failure Condition |
|:----:|-------|-------------------|
| 1 | Expiration | `expires_at` is in the past (default TTL: 90 days) |
| 2 | Environment/Scope | Requested target not in `approved_environments` or `approved_scopes` |
| 3 | Quorum | `len(approver_signatures) < required_approvers` |
| 4 | Store confirmation | `store.is_approved()` returns False (catches revocations) |
| 5 | Signature | Ed25519 signature verification fails (optional, requires public key) |

Any gate failure raises a descriptive `ValueError` identifying the failing gate and its specific condition. Gates are checked sequentially; the first failure halts promotion.

The `ModelApproval` dataclass captures the complete governance decision including `approval_id`, `model_id`, `weights_hash`, `approved_by`, `approved_at`, `expires_at`, `signature`, `status` (active/revoked), `approved_environments`, `approved_scopes`, `required_approvers`, and `approver_signatures` (approver ID to signature map).

## Key Design Decisions

- **Type-enforced separation (architectural, not just procedural):** `deploy_approved()` requires a `ModelApproval`, which can only come from `submit_for_governance()`, which requires a `TrainingOutput` from `complete_training()`. The type system makes single-path train-approve-deploy structurally difficult, going beyond policy-level or documentation-level controls. No single function in the public API spans all three planes.

- **Lazy crypto imports for zero-dep core:** Ed25519 signing requires the `cryptography` package, but it is imported lazily via `_require_cryptography()` so the core governance logic (TTL, quorum, scope, store confirmation) works with only the Python standard library. Users who do not need cryptographic signing can use the full governance framework without installing any optional dependencies.

- **Custom KS statistic (no scipy):** A two-pointer algorithm sweeps through sorted, normalized distributions to compute the maximum empirical CDF difference in O(n log n) time with O(n) space. Both distributions are normalized to [0, 1] before comparison. While it does not compute a p-value like scipy's `ks_2samp`, the configurable threshold (default 0.10) provides practical precision for production drift monitoring without scientific computing dependencies.

- **Hash excludes signatures (non-sequential signing):** `ModelApproval.compute_hash()` excludes `signature`, `status`, and `approver_signatures` from the canonical SHA-256 hash. This ensures all M-of-N approvers sign the same canonical content regardless of signing order. It also means revocation does not invalidate existing signatures, preserving them for audit purposes.

- **Auto-revocation on severe drift:** When drift severity reaches 0.8 or higher and `auto_revoke=True`, the coordinator automatically revokes the approval with `revoked_by="system:drift-detector"`, sets the approval status to `"revoked"` in the store, and records the event in the evidence ledger. Subsequent deployment attempts fail at Gate 4 (store confirmation).

## Ecosystem Integration

The `sm-model-governance` package occupies the governance layer in the NANDA ecosystem, providing cryptographic accountability for model deployment decisions.

| Package | Role | Question Answered |
|---------|------|-------------------|
| `sm-model-provenance` | Identity metadata | Where did this model come from? |
| `sm-model-card` | Metadata schema | What is this model? |
| `sm-model-integrity-layer` | Integrity verification | Does metadata meet policy? |
| **`sm-model-governance`** | **Cryptographic governance** | **Has this model been approved?** |
| `sm-bridge` | Transport layer | How is it exposed to the network? |

The model card's `weights_hash` becomes the integrity anchor in `ModelApproval`, linking governance decisions to specific trained weights. The bridge module provides `approval_to_integrity_facts()` to convert approvals into integrity-layer-compatible metadata and `create_provenance_with_approval()` to establish bidirectional links between governance and provenance.

A complete NANDA governance workflow spans all ecosystem packages: a `ModelCard` is created with model metadata and weights hash; the integrity layer verifies the hash, builds the lineage chain, creates an HMAC attestation, and runs governance policies; the `GovernanceCoordinator` receives the training output, obtains M-of-N Ed25519 signatures, and stores the time-bounded approval; the 5-gate promotion check validates all conditions before deployment; and drift detection continuously monitors production behavior, auto-revoking approvals on severe degradation.

The `ApprovalStore` protocol supports two backends: `InMemoryApprovalStore` (thread-safe via `threading.Lock()`, suitable for testing and single-process deployments) and `PostgresApprovalStore` (uses connection pooling, creates its schema on initialization, supports UPSERT, and provides `list_expiring(within_days)` for operational monitoring). Both implementations independently verify all five approval conditions in their `is_approved()` method, providing defense-in-depth against tampered approval objects.

The drift detection system supports both distribution drift (via the custom KS statistic) and metric-based drift (configurable thresholds for loss increase, accuracy ratio, and confidence). Severity is computed on a 0.0-1.0 scale and mapped to action recommendations: monitor (< 0.3), investigate (0.3-0.8), or consider_revoke (>= 0.8).

## References

1. NANDA Protocol. "Network of AI Agents in Decentralized Architecture." https://projectnanda.org
2. NIST. "Recommendation for Key Management." NIST SP 800-57. https://csrc.nist.gov/pubs/sp/800-57-pt1/r5/final

---

*First published: 2026-04-15 | Last modified: 2026-04-15*

*[stellarminds.ai](https://stellarminds.ai) — Research Contribution to [Project NANDA](https://projectnanda.org)*
