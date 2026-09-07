"""Shared fixtures - the fast tests run without torch or transformers."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

os.environ.setdefault("API_KEYS", "test-key,zweiter-key")
os.environ.setdefault("MAX_CONCURRENCY", "2")
os.environ.setdefault("NUM_THREADS", "1")

API_KEY = os.environ["API_KEYS"].split(",")[0]
SECOND_API_KEY = os.environ["API_KEYS"].split(",")[1]
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class FakePipeline:
    """Stands in for the HF pipeline: ascending scores, i.e. unsorted."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        text: str,
        candidate_labels: List[str],
        multi_label: bool,
        hypothesis_template: str,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "text": text,
                "candidate_labels": list(candidate_labels),
                "multi_label": multi_label,
                "hypothesis_template": hypothesis_template,
            }
        )
        weights = [i + 1 for i in range(len(candidate_labels))]
        total = sum(weights)
        return {
            "sequence": text,
            "labels": list(candidate_labels),
            "scores": [w / total for w in weights],
        }


@pytest.fixture
def fake_pipe(monkeypatch) -> FakePipeline:
    """Patches Classifier.load so no real model is loaded."""
    from app import classifier as classifier_module

    pipe = FakePipeline()

    def fake_load(self, settings) -> None:
        self._pipe = pipe

    monkeypatch.setattr(classifier_module.Classifier, "load", fake_load)
    yield pipe
    classifier_module.classifier._pipe = None


@pytest.fixture
def client(fake_pipe):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
