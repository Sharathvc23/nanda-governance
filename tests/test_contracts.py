"""Tests for protocol definitions in contracts.py."""

from __future__ import annotations

from typing import Any

from nanda_governance.contracts import (
    AdapterRegistry,
    EvidenceLedger,
    ModelCard,
    ModelValidator,
    ServingEndpoint,
    TrainingResult,
    ValidationResult,
)


# ------------------------------------------------------------------
# Concrete implementations for protocol conformance testing
# ------------------------------------------------------------------


class _StubModelCard:
    def __init__(self) -> None:
        self._model_id = "m1"
        self._weights_hash = "abc123"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self._model_id}


class _StubTrainingResult:
    @property
    def metrics(self) -> dict[str, Any]:
        return {"loss": 0.5}

    @property
    def model_id(self) -> str:
        return "m1"


class _StubRegistry:
    async def register(self, metadata: Any) -> Any:
        return metadata

    async def get(self, adapter_id: str, **kwargs: Any) -> Any:
        return None


class _StubLedger:
    def record(self, entry_type: str, data: dict[str, Any]) -> str:
        return "entry-1"


class _StubEndpoint:
    async def deploy(self, model_id: str, **kwargs: Any) -> bool:
        return True

    async def undeploy(self, model_id: str, **kwargs: Any) -> bool:
        return True


class _StubValidator:
    def validate(
        self,
        training_result: Any,
        model_card: Any,
        *,
        profile: str,
    ) -> ValidationResult:
        return ValidationResult(valid=True)


# ------------------------------------------------------------------
# Protocol conformance tests
# ------------------------------------------------------------------


def test_model_card_protocol() -> None:
    card = _StubModelCard()
    assert isinstance(card, ModelCard)
    assert card.model_id == "m1"
    assert card.weights_hash == "abc123"
    assert "model_id" in card.to_dict()


def test_training_result_protocol() -> None:
    result = _StubTrainingResult()
    assert isinstance(result, TrainingResult)
    assert result.model_id == "m1"
    assert "loss" in result.metrics


def test_adapter_registry_protocol() -> None:
    reg = _StubRegistry()
    assert isinstance(reg, AdapterRegistry)


def test_evidence_ledger_protocol() -> None:
    ledger = _StubLedger()
    assert isinstance(ledger, EvidenceLedger)
    entry_id = ledger.record("test", {"key": "value"})
    assert entry_id == "entry-1"


def test_serving_endpoint_protocol() -> None:
    ep = _StubEndpoint()
    assert isinstance(ep, ServingEndpoint)


def test_model_validator_protocol() -> None:
    v = _StubValidator()
    assert isinstance(v, ModelValidator)
    result = v.validate(
        _StubTrainingResult(), _StubModelCard(), profile="default"
    )
    assert result.valid is True


def test_validation_result_frozen() -> None:
    r = ValidationResult(valid=False, message="bad")
    assert r.valid is False
    assert r.message == "bad"
