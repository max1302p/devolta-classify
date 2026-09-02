"""FastAPI application: routes, lifespan, auth, error handling."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .classifier import ModelNotLoadedError, Prediction, classifier
from .config import Settings, get_settings
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    LabelScore,
)

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
    version="1.0.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)


# --------------------------------------------------------------------------- #
# Error handling: every failure comes back as {"error": "..."}
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err.get("loc", ()) if part != "body")
        message = err.get("msg", "invalid value")
        # Pydantic prefixes custom validators with "Value error, " - just noise.
        for prefix in ("Value error, ", "Assertion failed, "):
            if message.startswith(prefix):
                message = message[len(prefix):]
        parts.append(f"{location}: {message}" if location else message)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "; ".join(parts) or "invalid request"},
    )


@app.exception_handler(HTTPException)
async def _http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "model is not loaded yet"},
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
