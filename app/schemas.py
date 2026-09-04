"""Pydantic v2 models for requests and responses."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LABELS = 50
MIN_LABELS = 2
MAX_TEXT_LEN = 5000
MAX_BATCH_ITEMS = 32


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LEN,
        description="Text to classify (1-5000 characters).",
        examples=["Mein Kassensystem druckt seit gestern keine Bons mehr"],
    )
    labels: List[str] = Field(
        ...,
        min_length=MIN_LABELS,
        max_length=MAX_LABELS,
        description="2-50 unique, non-empty labels.",
        examples=[["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"]],
    )
    multi_label: bool = Field(
        default=False,
        description="True scores every label independently.",
    )
    hypothesis_template: Optional[str] = Field(
        default=None,
        description=(
            "Hypothesis template, must contain '{}'. Defaults to the "
            "DEFAULT_HYPOTHESIS_TEMPLATE environment variable."
        ),
        examples=["Diese Nachricht betrifft {}."],
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be only whitespace")
        return value

    @field_validator("labels")
    @classmethod
    def _labels_clean_and_unique(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for label in value:
            stripped = label.strip()
            if not stripped:
                raise ValueError("labels must not be empty")
            cleaned.append(stripped)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("labels must be unique")
        return cleaned

    @field_validator("hypothesis_template")
    @classmethod
    def _template_has_placeholder(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if "{}" not in value:
            raise ValueError("hypothesis_template must contain '{}'")
        return value

    def template(self, default: str) -> str:
        return self.hypothesis_template or default


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[ClassifyRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_ITEMS,
        description=f"1-{MAX_BATCH_ITEMS} items.",
    )


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    label: str = Field(description="Label with the highest score.")
    score: float = Field(description="Score of the top label.")
    results: List[LabelScore] = Field(description="All labels, highest score first.")
    multi_label: bool
    model: str
    duration_ms: int


class BatchResponse(BaseModel):
    results: List[ClassifyResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    error: str
