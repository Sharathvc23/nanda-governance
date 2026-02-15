# Three-Plane Separation for Cryptographic ML Model Governance

**Authors:** StellarMinds ([stellarminds.ai](https://stellarminds.ai))
**Date:** February 2026
**Version:** 1.0

## Abstract

Deploying machine learning models in production requires governance guarantees that no single actor or code path can unilaterally train, approve, and deploy a model. Many existing ML deployment frameworks bundle these concerns into unified pipelines, which can create single points of failure for both security and compliance. This paper presents `nanda-model-governance`, a Python library that enforces three-plane separation — training, governance, and serving — through type-enforced handoffs and cryptographic approval gates. The system provides Ed25519 digital signatures with lazy imports for zero-dependency core operation, a 5-gate promotion check that validates expiration, scope, quorum, store confirmation, and signature before deployment, a custom two-sample Kolmogorov-Smirnov statistic for scipy-free drift detection, M-of-N multi-approver quorum with per-approver signatures, time-bounded approvals with configurable TTL (default 90 days), automatic approval revocation on severe drift, and a pluggable `ApprovalStore` protocol with in-memory and PostgreSQL backends. The implementation requires no runtime dependencies for core operation, passes strict static analysis, and integrates with the NANDA ecosystem for integrity verification and agent discovery.

## 1. Introduction

### 1.1 Problem Statement

In many conventional ML deployment pipelines, a single workflow — often a CI/CD job or notebook — can train a model, evaluate its metrics, and push it to production. This conflation of responsibilities can create several risks:

- **Insider threat** — A compromised or malicious actor with pipeline access can substitute model weights without independent review.
- **Compliance violations** — Regulatory frameworks (SOX, NIST SP 800-57) require separation of duties for critical operations, but monolithic ML pipelines lack formal separation boundaries.
- **Stale deployments** — Without time-bounded approvals, a model approved months ago can be deployed without re-evaluation, even as the data distribution has shifted.

### 1.2 Motivation

The NANDA ecosystem requires a governance layer that:

1. **Enforces structural separation** — making it architecturally difficult for a single code path to span training, approval, and deployment.
2. **Provides cryptographic accountability** — every approval is signed and attributable to a specific identity.
3. **Supports multi-party governance** — critical models require M-of-N approvals before deployment.
4. **Detects and responds to drift** — production models that diverge from training characteristics are flagged and optionally revoked.
5. **Maintains zero core dependencies** — the governance framework operates with only the Python standard library, with optional extras for cryptography and database backends.

### 1.3 Contributions

This paper makes the following contributions:

- A **three-plane governance architecture** with type-enforced handoffs (`TrainingOutput → ModelApproval → PromotionResult`) that strongly discourages any single execution path from spanning all three planes.
- A **5-gate promotion check** that sequentially validates expiration, environment/scope, quorum, store confirmation, and cryptographic signature before allowing deployment.
- **Ed25519 digital signing** with lazy imports, maintaining zero runtime dependencies while providing asymmetric cryptographic guarantees when the `cryptography` package is available.
- An **M-of-N quorum mechanism** where each approver contributes an independent Ed25519 signature over a deterministic canonical hash.
- A **custom two-sample Kolmogorov-Smirnov implementation** that detects distribution drift without requiring scipy, using a two-pointer algorithm over sorted normalized distributions.
- **Time-bounded approvals** with configurable TTL (default 90 days) and automatic revocation on severe drift (severity ≥ 0.8).
- A **pluggable `ApprovalStore` protocol** with thread-safe in-memory and PostgreSQL-backed implementations.

## 2. Related Work

### 2.1 Separation of Duties in Information Security

The principle of separation of duties (SoD) is foundational to information security. NIST SP 800-57 (Recommendation for Key Management) mandates split-knowledge procedures for cryptographic key management, ensuring no single individual possesses complete key material. SOX (Sarbanes-Oxley Act) Section 404 requires internal controls over financial reporting, including segregation of duties. The COBIT framework operationalizes SoD through control objectives for IT governance. This work applies SoD principles specifically to the ML model lifecycle, using typed handoffs and cryptographic signatures as the separation mechanism.

### 2.2 MLflow Model Registry

MLflow's model registry provides model versioning and stage transitions (Staging → Production → Archived) with optional review workflows. However, MLflow does not currently provide several governance properties addressed here: transitions can typically be performed by any user with write access (without multi-party approval), there is no built-in cryptographic signing of approvals, and there is no native mechanism for time-bounded approval validity or automatic revocation on drift.

### 2.3 Seldon Deploy

Seldon Deploy provides a Kubernetes-native model deployment platform with canary deployments, A/B testing, and monitoring. While Seldon includes drift detection (via Alibi Detect), it operates primarily as an infrastructure layer rather than a governance framework — it does not currently enforce formal separation between who trains and who deploys, nor does it provide multi-signature approval workflows or time-bounded approval semantics.

### 2.4 Kubernetes Admission Controllers

Kubernetes admission controllers (ValidatingWebhookConfiguration) provide a mechanism for enforcing policies on resource creation. While conceptually similar to this work's promotion gates, admission controllers operate at the infrastructure level on generic Kubernetes resources rather than understanding ML-specific governance concepts like training metrics, quorum requirements, or drift detection.

### 2.5 Gaps Addressed

This work addresses four gaps in the existing landscape:

1. **Typed three-plane separation** — architecturally enforced separation of training, governance, and serving through distinct types and APIs.
2. **Multi-party cryptographic governance** — M-of-N Ed25519 signatures with per-approver accountability.
3. **Scipy-free drift detection** — custom KS statistic implementation enabling governance-aware drift monitoring without heavy scientific computing dependencies.
4. **Time-bounded, auto-revocable approvals** — approvals that expire and can be automatically revoked when production behavior deviates from training baselines.

## 3. Design / Architecture

### 3.1 Three-Plane Separation

The architecture partitions the model lifecycle into three isolated planes, each with a distinct API, output type, and audit event:

```
 Training Plane          Governance Plane           Serving Plane
      │                       │                         │
 complete_training()    submit_for_governance()    deploy_approved()
      │                       │                         │
      ▼                       ▼                         ▼
 TrainingOutput ──────► ModelApproval ──────────► PromotionResult
                         (signed,
                          scoped,
                          time-bounded)
```

**Training Plane** — Produces a `TrainingOutput` record containing the model identifier, weights hash, training metrics, and a correlation ID for audit tracing. This plane has no knowledge of approval semantics or deployment targets.

**Governance Plane** — Accepts a `TrainingOutput` and produces a `ModelApproval` record. This plane handles signing, quorum tracking, scope constraints, and TTL assignment. It stores the approval in an `ApprovalStore` backend.

**Serving Plane** — Accepts a `ModelApproval` and, after passing all five promotion gates, produces a `PromotionResult`. This plane can optionally call a `ServingEndpoint` to execute the actual deployment.

The separation is encouraged through **type-level constraints**: `deploy_approved()` requires a `ModelApproval` parameter, which is normally obtained from `submit_for_governance()`, which in turn requires a `TrainingOutput` from `complete_training()`. No single function in the public API spans all three planes.

### 3.2 The GovernanceCoordinator

The `GovernanceCoordinator` class orchestrates the three-plane flow while maintaining the separation invariant. It accepts four optional protocol-typed dependencies:

| Dependency | Protocol | Purpose |
|------------|----------|---------|
| `store` | `ApprovalStore` | Approval persistence (default: `InMemoryApprovalStore`) |
| `ledger` | `EvidenceLedger` | Audit trail recording |
| `validator` | `ModelValidator` | Optional pre-governance validation |
| `endpoint` | `ServingEndpoint` | Optional deployment execution |

All dependencies are protocol-typed, allowing injection of custom implementations without inheritance.

### 3.3 Pluggable Protocol Contracts

The library defines seven `@runtime_checkable` protocols:

| Protocol | Methods | Plane |
|----------|---------|-------|
| `ApprovalStore` | `store()`, `get()`, `is_approved()`, `revoke()` | Governance / Serving |
| `EvidenceLedger` | `record()` | All planes |
| `ModelValidator` | `validate()` | Governance |
| `ServingEndpoint` | `deploy()`, `undeploy()` | Serving |
| `ModelCard` | `model_id`, `weights_hash`, `to_dict()` | Training |
| `TrainingResult` | `model_id`, `metrics` | Training |
| `AdapterRegistry` | `register()`, `get()` | Integration |

### 3.4 Approval Data Model

The `ModelApproval` dataclass captures the complete governance decision:

| Field | Type | Purpose |
|-------|------|---------|
| `approval_id` | `str` | Auto-generated unique ID |
| `model_id` | `str` | Model being approved |
| `weights_hash` | `str` | Integrity anchor to training output |
| `approved_by` | `str` | Primary approver identity |
| `approved_at` | `datetime` | Approval timestamp (UTC) |
| `expires_at` | `datetime \| None` | TTL expiration (default: now + 90 days) |
| `profile` | `str` | Deployment profile |
| `correlation_id` | `str` | Audit trace ID |
| `signature` | `str` | Ed25519 signature (hex) |
| `status` | `str` | active, revoked |
| `approved_environments` | `list[str]` | Allowed deployment targets |
| `approved_scopes` | `list[str]` | Allowed scope constraints |
| `required_approvers` | `int` | M in M-of-N quorum |
| `approver_signatures` | `dict[str, str]` | Approver ID → signature map |

## 4. Implementation

### 4.1 Ed25519 Signing with Lazy Imports

The signing module avoids a hard dependency on the `cryptography` package through lazy imports:

```python
def _require_cryptography() -> None:
    from nanda_governance._compat import has_cryptography
    if not has_cryptography():
        raise ImportError(
            "Ed25519 signing requires the 'cryptography' package. "
            "Install it with: pip install nanda-model-governance[crypto]"
        )
```

The `sign_approval()` function:

1. Validates `cryptography` availability.
2. Computes the approval's canonical hash via `ModelApproval.compute_hash()`.
3. Signs the hash digest with the provided `Ed25519PrivateKey`.
4. Returns the hex-encoded signature.

The `verify_approval()` function reverses this process, recomputing the hash and verifying the signature. Verification catches both `InvalidSignature` and `ValueError` exceptions, returning `False` for any failure without leaking error details.

The `compute_hash()` method produces a deterministic SHA-256 digest of the approval's immutable fields, **excluding** `signature`, `status`, and `approver_signatures`. This ensures all approvers sign the same hash regardless of how many signatures have been collected, and that revocation does not invalidate existing signatures for audit purposes.

### 4.2 Five-Gate Promotion Check

The `promote_model()` async function implements five sequential validation gates. All gates must pass; any failure raises a `ValueError` with a descriptive message identifying the failing gate.

**Gate 1: Expiration Check**

```python
if approval.is_expired():
    raise ValueError(f"Approval for model {approval.model_id} has expired "
                     f"(expired at {approval.expires_at})")
```

The `is_expired()` method compares `datetime.now(timezone.utc)` against `expires_at`. Approvals with `expires_at = None` never expire.

**Gate 2: Environment/Scope Validation**

```python
if not approval.is_valid_for(environment, scope):
    raise ValueError(...)
```

The `is_valid_for()` method checks that the requested deployment environment appears in `approved_environments` (if set) and the requested scope appears in `approved_scopes` (if set). Unconstrained approvals (empty lists) pass any environment or scope.

**Gate 3: Quorum Verification**

```python
if not approval.has_quorum():
    raise ValueError(...)
```

The `has_quorum()` method returns `True` when `len(approver_signatures) >= required_approvers`. This implements M-of-N governance: an approval requiring 2 signatures blocks deployment until at least 2 distinct approvers have signed.

**Gate 4: Store Confirmation**

```python
if not store.is_approved(model_id, environment, scope):
    raise ValueError(...)
```

This gate double-checks the authoritative store, preventing replay attacks where an approval object is presented after having been revoked in the store. The store's `is_approved()` method independently verifies status, expiration, scope, and quorum.

**Gate 5: Signature Verification (Optional)**

```python
if public_key is not None:
    if not verify_approval(approval, public_key):
        raise ValueError(...)
```

When a public key is provided, the gate verifies the Ed25519 signature cryptographically. This gate is optional to support store-only deployment environments where cryptographic verification is performed at a different layer.

### 4.3 Custom Kolmogorov-Smirnov Statistic

The `check_distribution_drift()` function implements a two-sample KS test without scipy, suitable for detecting distribution shifts between training and production outputs.

**Algorithm:**

1. **Minimum sample check** — Requires at least 10 samples from each distribution. Returns a low-confidence result if insufficient.

2. **Normalization** — Both distributions are rescaled to [0, 1]:
   ```python
   range_val = max_val - min_val if max_val > min_val else 1.0
   training_norm = sorted((v - min_val) / range_val for v in training_outputs)
   serving_norm = sorted((v - min_val) / range_val for v in serving_outputs)
   ```

3. **Two-pointer CDF comparison** — The algorithm sweeps through both sorted distributions simultaneously, computing the empirical CDF difference at each step:
   ```python
   i, j = 0, 0
   while i < n_train and j < n_serve:
       if training_norm[i] <= serving_norm[j]:
           diff = abs((i + 1) / n_train - j / n_serve)
           i += 1
       else:
           diff = abs(i / n_train - (j + 1) / n_serve)
           j += 1
       max_diff = max(max_diff, diff)
   ```

4. **Threshold comparison** — The KS statistic is compared against a configurable threshold (default 0.10):
   ```python
   is_drifted = ks_statistic > config.ks_threshold
   ```

The algorithm runs in O(n log n) time (dominated by sorting) with O(n) space. While it does not compute a p-value (as scipy's `ks_2samp` does), the configurable threshold provides practical precision for many production drift monitoring scenarios.

### 4.4 Metric-Based Drift Detection

Beyond distribution drift, the `check_drift()` function supports metric-based drift detection through `DriftConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_loss_increase` | 0.20 | Maximum tolerated loss increase (20%) |
| `min_accuracy_ratio` | 0.95 | Minimum serving/training accuracy ratio |
| `confidence_threshold` | 0.90 | Confidence threshold for alerting |
| `ks_threshold` | 0.10 | KS statistic threshold |

Severity is computed on a 0.0–1.0 scale and mapped to action recommendations:

| Severity Range | Recommended Action |
|:--------------:|:------------------:|
| 0.0 | monitor |
| < 0.3 | monitor |
| 0.3–0.6 | investigate |
| 0.6–0.8 | investigate |
| ≥ 0.8 | consider_revoke |

### 4.5 M-of-N Quorum

The quorum mechanism supports multi-party governance through independent signatures:

1. **Initial approval** — The primary approver submits governance with `required_approvers=M` and their private key. The coordinator signs and stores the first signature in `approver_signatures`.

2. **Additional signatures** — Each subsequent approver calls `add_approval_signature(model_id, approver_id, private_key)`. The coordinator computes the same deterministic hash (since it excludes signatures from the hash input) and stores the new signature.

3. **Quorum check** — `has_quorum()` returns `True` when `len(approver_signatures) >= required_approvers`.

The hash exclusion of `approver_signatures` is critical: it ensures all approvers sign the same canonical content, regardless of signing order. This property also allows signature verification to be performed independently for each approver using their respective public key.

### 4.6 Time-Bounded Approvals and Auto-Revocation

**TTL assignment** occurs at governance submission:

```python
expires_at = None
if approval_ttl_days is not None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=approval_ttl_days)
```

The default TTL of 90 days (`DEFAULT_APPROVAL_TTL_DAYS = 90`) provides a balance between operational convenience and stale-approval risk. Setting `approval_ttl_days=None` disables expiration.

**Auto-revocation** is triggered by the `check_drift()` method when three conditions are met simultaneously:

1. `auto_revoke=True` (explicitly enabled by the caller).
2. `result.is_drifted is True` (drift was detected).
3. `result.overall_severity >= 0.8` (severity is critical).

Upon triggering, the coordinator calls `revoke_model()` with `revoked_by="system:drift-detector"`, setting the approval's `status` to `"revoked"` in the store and recording the event in the evidence ledger. Subsequent deployment attempts fail at Gate 4 (store confirmation).

### 4.7 ApprovalStore Backends

The `ApprovalStore` protocol defines four methods:

```python
@runtime_checkable
class ApprovalStore(Protocol):
    def store(self, approval: ModelApproval) -> None: ...
    def get(self, model_id: str) -> ModelApproval | None: ...
    def is_approved(self, model_id: str, *,
                    environment: str | None = None,
                    scope: str | None = None) -> bool: ...
    def revoke(self, model_id: str, revoked_by: str, reason: str) -> bool: ...
```

**InMemoryApprovalStore** — Thread-safe via `threading.Lock()`, suitable for testing and single-process deployments. Data is lost on process restart.

**PostgresApprovalStore** — Uses `psycopg2.pool.ThreadedConnectionPool` for connection management. Creates its schema (table and indexes) on initialization. Supports UPSERT via `ON CONFLICT ... DO UPDATE`. Provides an additional `list_expiring(within_days)` method for operational monitoring. The schema includes indexes on `model_id`, `status`, and `expires_at` for efficient queries.

Both implementations independently verify all five approval conditions in their `is_approved()` method (status, expiration, environment, scope, quorum), providing an additional layer of defense against tampered approval objects.

## 5. Integration

### 5.1 NANDA Ecosystem Context

The `nanda-model-governance` package occupies the **governance layer** in the NANDA ecosystem, answering the question: *"Has this model been approved?"*

| Package | Role | Question Answered |
|---------|------|-------------------|
| `nanda-model-card` | Metadata schema | What is this model? |
| `nanda-model-integrity-layer` | Integrity verification | Does this model's metadata meet policy? |
| `nanda-model-governance` | Cryptographic governance | Has this model been approved? |

### 5.2 Integration with the Model Card Schema

The governance coordinator's `complete_training()` method accepts a `card` dictionary parameter, allowing the full `ModelCard` (from `nanda-model-card`) to accompany the training output through the governance pipeline. The `weights_hash` field from the model card becomes the integrity anchor in `ModelApproval`, linking the governance decision to a specific set of trained weights.

The governance layer's `ModelCard` protocol mirrors the key fields of the `nanda-model-card` `ModelCard` dataclass (`model_id`, `weights_hash`, `to_dict()`), enabling type-safe integration without creating a direct package dependency.

### 5.3 Integration with the Integrity Layer

The bridge module (`nanda.py`) provides two integration functions:

- **`approval_to_integrity_facts(approval)`** — Converts a `ModelApproval` into a dictionary suitable for embedding in the integrity layer's `IntegrityExtension`, including `model_id`, `weights_hash`, `approved_by`, `approved_at`, `status`, and `correlation_id`.
- **`create_provenance_with_approval(approval, model_name, model_version)`** — Creates a `ModelProvenance` record (from `nanda-model-integrity-layer`) populated with governance metadata, establishing a bidirectional link between the governance and integrity layers.

These functions handle the optional dependency gracefully: if `nanda-model-integrity-layer` is not installed, they raise an `ImportError` with installation instructions.

### 5.4 End-to-End Workflow

A complete NANDA governance workflow spans all three packages:

1. **Model Card Creation** — A `ModelCard` (from `nanda-model-card`) is created with model metadata, weights hash, and lifecycle status.
2. **Integrity Verification** — The `nanda-model-integrity-layer` verifies the weights hash, builds the lineage chain, creates an HMAC attestation, and runs governance policies.
3. **Governance Approval** — The `GovernanceCoordinator` receives the training output, obtains M-of-N Ed25519 signatures, and stores the time-bounded approval.
4. **Promotion** — The 5-gate promotion check validates all conditions before deployment.
5. **Monitoring** — Drift detection continuously monitors production behavior and auto-revokes approvals on severe degradation.

## 6. Evaluation

### 6.1 Test Coverage

The test suite contains **76 test methods** across seven test modules:

| Test Module | Tests | Coverage Area |
|-------------|:-----:|---------------|
| `test_approval.py` | 14 | Defaults, hash determinism, expiration, scope, quorum, serialization |
| `test_signing.py` | 7 | Sign/verify round-trip, tamper detection, wrong key, missing crypto |
| `test_promotion.py` | 7 | All 5 gates individually, evidence recording |
| `test_stores.py` | 15 | CRUD, expiration, scope, quorum, revocation for both backends |
| `test_contracts.py` | 7 | All protocol conformance checks, immutability |
| `test_coordinator.py` | 17 | Full three-plane flow, multi-approver quorum, revocation, drift |
| `test_drift.py` | 15 | Metric drift, distribution drift, severity levels, alerting |

Tests requiring the `cryptography` package use a `skip_no_crypto` marker, ensuring the core test suite passes without optional dependencies. Async tests (deployment flow) use `pytest-asyncio`.

### 6.2 Example: Three-Plane Governance Flow

```python
import asyncio
from nanda_governance import GovernanceCoordinator

async def main():
    coord = GovernanceCoordinator()

    # Training Plane
    output = coord.complete_training(
        model_id="sentiment-v3",
        weights_hash="sha256:abcdef1234567890",
        metrics={"loss": 0.28, "accuracy": 0.94},
    )

    # Governance Plane (unsigned, single approver)
    approval = coord.submit_for_governance(
        output,
        approved_by="governance-lead",
        approved_environments=["staging", "production"],
        approval_ttl_days=90,
    )

    # Serving Plane
    result = await coord.deploy_approved(
        approval, environment="staging"
    )
    assert result.promoted is True

asyncio.run(main())
```

### 6.3 Example: Multi-Approver Quorum with Drift Monitoring

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from nanda_governance import GovernanceCoordinator, DriftConfig

coord = GovernanceCoordinator()
key_alice = Ed25519PrivateKey.generate()
key_bob = Ed25519PrivateKey.generate()

# Training
output = coord.complete_training("model-v1", "sha256:abc", {"loss": 0.10})

# Governance — require 2-of-N approval
approval = coord.submit_for_governance(
    output, approved_by="alice", private_key=key_alice, required_approvers=2
)
assert not coord.store.is_approved("model-v1")  # Only 1/2

coord.add_approval_signature("model-v1", "bob", key_bob)
assert coord.store.is_approved("model-v1")       # Now 2/2

# Drift monitoring with auto-revocation
result = coord.check_drift(
    "model-v1",
    training_metrics={"loss": 0.10},
    serving_metrics={"loss": 2.50},  # 25x increase
    auto_revoke=True,
)
assert result.is_drifted
assert not coord.store.is_approved("model-v1")  # Auto-revoked
```

### 6.4 Gate Failure Behavior

Each promotion gate produces a specific, actionable error:

| Gate | Trigger | Error Message Pattern |
|------|---------|----------------------|
| 1 | TTL expired | `"Approval for model X has expired (expired at ...)"` |
| 2 | Wrong environment | `"Approval for model X not valid for environment/scope"` |
| 3 | Insufficient signatures | `"Approval for model X lacks quorum (1/2 signatures)"` |
| 4 | Store says revoked | `"Model X not approved in store"` |
| 5 | Bad signature | `"Invalid signature for model X"` |

## 7. Conclusion

### 7.1 Summary

This paper presented `nanda-model-governance`, a library that brings separation-of-duties principles from information security to the ML model lifecycle. By partitioning the lifecycle into three typed planes with cryptographic handoffs, the architecture makes it significantly harder for a single code path to train, approve, and deploy a model. The 5-gate promotion check provides defense-in-depth, the M-of-N quorum supports multi-party governance, and the time-bounded approval mechanism with auto-revocation on drift helps prevent stale models from remaining in production. The zero-dependency core with lazy optional imports suggests that robust governance need not come at the cost of deployment complexity.

### 7.2 Future Work

Several directions merit investigation:

- **Hardware Security Module (HSM) integration** — Implementing the `Signer` protocol against PKCS#11 interfaces for enterprise key management.
- **Approval delegation and escalation** — Supporting delegated signing authority (e.g., "Alice delegates to Bob for 48 hours") and automatic escalation paths when quorum is not met within a deadline.
- **Continuous governance** — Extending the TTL mechanism to support approval renewal workflows where a model can be re-approved without re-training.
- **Cross-registry governance** — Enabling governance decisions made in one NANDA registry to be verifiable in another through portable approval records.
- **Federated quorum** — Supporting quorum across organizational boundaries, where approvers from different entities contribute signatures to a shared governance decision.

## References

1. National Institute of Standards and Technology. (2020). "Recommendation for Key Management: Part 1 — General." NIST SP 800-57 Part 1, Revision 5.

2. U.S. Congress. (2002). "Sarbanes-Oxley Act of 2002." Public Law 107-204.

3. ISACA. "COBIT 2019 Framework: Governance and Management Objectives." Information Systems Audit and Control Association.

4. Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S.A., Konwinski, A., Murching, S., Nykodym, T., Ogilvie, P., Parkhe, M., Xie, F., and Zuber, C. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Engineering Bulletin*, 41(4), pp. 39–45.

5. Seldon Technologies. "Seldon Deploy: Enterprise Model Deployment and Monitoring." https://www.seldon.io

6. Bernstein, D.J., Duif, N., Lange, T., Schwabe, P., and Yang, B-Y. (2012). "High-speed high-security signatures." *Journal of Cryptographic Engineering*, 2(2), pp. 77–89.

7. Kolmogorov, A.N. (1933). "Sulla determinazione empirica di una legge di distribuzione." *Giornale dell'Istituto Italiano degli Attuari*, 4, pp. 83–91.

8. Smirnov, N.V. (1948). "Table for estimating the goodness of fit of empirical distributions." *Annals of Mathematical Statistics*, 19(2), pp. 279–281.

9. NANDA Protocol. "Network for Agent Discovery and Attestation." https://projectnanda.org

10. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., and Gebru, T. (2019). "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\*)*, pp. 220–229.
