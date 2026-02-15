# nanda-model-governance

Three-plane ML governance with cryptographic approvals for [NANDA](https://projectnanda.org)-compatible agent registries.

Enforces that no single execution path can train, approve, **and** deploy a model. Three isolated planes — Training, Governance, Serving — with Ed25519 cryptographic signatures, time-bounded approvals (90-day TTL), environment/scope constraints, M-of-N multi-approver quorum, drift detection, and revocation.

## Relationship to nanda-model-integrity-layer

| Package | Question it answers |
|---------|-------------------|
| [`nanda-model-integrity-layer`](https://github.com/Sharathvc23/nanda-model-integrity-layer) | "Does this model's metadata meet policy?" (rule-based checks) |
| `nanda-model-governance` (this package) | "Has this model been cryptographically approved for deployment?" (approval flow with signatures, quorum, scoping, revocation) |

## Installation

```bash
# Core (zero dependencies)
pip install nanda-model-governance

# With Ed25519 signing
pip install nanda-model-governance[crypto]

# With PostgreSQL store
pip install nanda-model-governance[postgres]

# With integrity layer bridge
pip install nanda-model-governance[integrity]

# Development
pip install nanda-model-governance[dev]
```

## Quick Start

```python
import asyncio
from nanda_governance import GovernanceCoordinator

async def main():
    coord = GovernanceCoordinator()

    # 1. Training Plane — produce a handoff object
    output = coord.complete_training(
        model_id="sentiment-v3",
        weights_hash="sha256:abcdef1234567890",
        metrics={"loss": 0.28, "accuracy": 0.94},
    )

    # 2. Governance Plane — create and store approval
    approval = coord.submit_for_governance(
        output,
        approved_by="governance-team",
        approved_environments=["staging", "production"],
        approval_ttl_days=90,
    )

    # 3. Serving Plane — verify approval and deploy
    result = await coord.deploy_approved(
        approval, environment="staging"
    )
    print(f"Deployed: {result.promoted}")

asyncio.run(main())
```

## Three-Plane Architecture

```
Training Plane          Governance Plane          Serving Plane
     |                       |                        |
complete_training()    submit_for_governance()   deploy_approved()
     |                       |                        |
TrainingOutput ──────> ModelApproval ──────────> PromotionResult
                        (signed, scoped,
                         time-bounded)
```

Each plane produces an output that becomes the next plane's input. No single code path can bypass the governance gate because:

- **Training** produces a `TrainingOutput` (model identity + weights hash)
- **Governance** validates, signs, and stores a `ModelApproval` (with Ed25519 signature, TTL, environment/scope constraints, and M-of-N quorum)
- **Serving** verifies the approval against the store before deployment

## Features

### Cryptographic Signing (Ed25519)

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from nanda_governance import sign_approval, verify_approval, ModelApproval

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()

approval = ModelApproval(model_id="m1", approved_by="alice")
approval.signature = sign_approval(approval, private_key)
assert verify_approval(approval, public_key)
```

### Multi-Approver Quorum

```python
approval = coord.submit_for_governance(
    output,
    approved_by="alice",
    private_key=key_alice,
    required_approvers=2,  # Need 2 signatures
)

# Add second signature
coord.add_approval_signature("model-id", "bob", key_bob)
```

### Drift Detection

```python
result = coord.check_drift(
    "model-id",
    training_metrics={"loss": 0.25, "accuracy": 0.93},
    serving_metrics={"loss": 0.45, "accuracy": 0.80},
    auto_revoke=True,  # Revoke on severe drift
)
```

### Pluggable Backends

All external dependencies are `@runtime_checkable Protocol` classes:

- `ApprovalStore` — persistence backend (in-memory or PostgreSQL included)
- `EvidenceLedger` — audit log
- `ServingEndpoint` — deployment target
- `ModelValidator` — governance gate checks
- `AdapterRegistry` — model registry

## API Reference

### GovernanceCoordinator

| Method | Plane | Description |
|--------|-------|-------------|
| `complete_training()` | Training | Produce a `TrainingOutput` handoff |
| `submit_for_governance()` | Governance | Validate, sign, and store approval |
| `add_approval_signature()` | Governance | Add signature for multi-approver quorum |
| `deploy_approved()` | Serving | Verify approval and deploy |
| `revoke_model()` | Governance | Revoke a model's approval |
| `check_drift()` | Monitoring | Check for model drift |

### Core Types

| Type | Description |
|------|-------------|
| `TrainingOutput` | Training plane exit handoff |
| `ModelApproval` | Signed approval with TTL, scope, quorum |
| `PromotionResult` | Deployment outcome |
| `DriftCheckResult` | Drift assessment |
| `DriftAlert` | Alert event for notification systems |

## Development

```bash
git clone https://github.com/Sharathvc23/nanda-model-governance.git
cd nanda-model-governance
pip install -e ".[dev,crypto]"
pytest tests/ -v
ruff check nanda_governance/
mypy nanda_governance/ --strict
```

## Credits

Developed by [stellarminds.ai](https://stellarminds.ai), open-sourced for [projectnanda.org](https://projectnanda.org).

## License

MIT
