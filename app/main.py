"""FastAPI application: routes, lifespan, auth, error handling."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .classifier import ModelNotLoadedError, Prediction, classifier
from .config import Settings, get_settings
from .schemas import ClassifyRequest, ClassifyResponse, HealthResponse, LabelScore

logger = logging.getLogger("classify.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # raises ConfigError when API_KEYS is missing

    logger.info(
        "startup",
        extra={
            "model": settings.model_id,
            "num_threads": settings.num_threads,
        },
    )
    await run_in_threadpool(classifier.load, settings)
    try:
        yield
    finally:
        logger.info("shutdown")


app = FastAPI(
    title="Zero-shot classification API",
    description=(
        "Zero-shot classification of German and English text using "
        "bge-m3-zeroshot-v2.0 on CPU."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> JSONResponse:
    loaded = classifier.is_loaded
    return JSONResponse(
        status_code=status.HTTP_200_OK if loaded else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ok" if loaded else "loading", "model_loaded": loaded},
    )


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    tags=["classify"],
)
async def classify(
    payload: ClassifyRequest,
    settings: Settings = Depends(get_settings),
) -> ClassifyResponse:
    _ensure_loaded()
    template = payload.template(settings.default_hypothesis_template)

    prediction = await run_in_threadpool(
        classifier.classify,
        payload.text,
        payload.labels,
        payload.multi_label,
        template,
    )

    return _to_response(prediction, payload.multi_label, settings)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_loaded() -> None:
    if not classifier.is_loaded:
        raise ModelNotLoadedError("model is not loaded")


def _to_response(
    prediction: Prediction, multi_label: bool, settings: Settings
) -> ClassifyResponse:
    top_label, top_score = prediction.top
    return ClassifyResponse(
        label=top_label,
        score=top_score,
        results=[LabelScore(label=label, score=score) for label, score in prediction.results],
        multi_label=multi_label,
        model=settings.model_id,
        duration_ms=prediction.duration_ms,
    )
