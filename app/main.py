"""FastAPI application: routes, lifespan, auth, error handling."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .classifier import ModelNotLoadedError, Prediction, classifier
from .config import Settings, get_settings
from .logging_config import configure_logging
from .schemas import (
    BatchRequest,
    BatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    HealthResponse,
    LabelScore,
)

logger = logging.getLogger("classify.api")

# Set during the lifespan; caps concurrent inferences (MAX_CONCURRENCY).
_inference_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()  # raises ConfigError when API_KEYS is missing
    configure_logging()

    global _inference_semaphore
    _inference_semaphore = asyncio.Semaphore(settings.max_concurrency)

    logger.info(
        "startup",
        extra={
            "model": settings.model_id,
            "num_threads": settings.num_threads,
            "max_concurrency": settings.max_concurrency,
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
        401: {"model": ErrorResponse},
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
# Auth
# --------------------------------------------------------------------------- #
def _extract_key(authorization: Optional[str], x_api_key: Optional[str]) -> str:
    """Prefers 'Authorization: Bearer <key>', accepts 'X-API-Key: <key>'."""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return (x_api_key or "").strip()


async def require_api_key(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    provided = _extract_key(authorization, x_api_key).encode("utf-8")
    # Compare against every key without an early exit, to keep the timing flat.
    matched = False
    for key in settings.api_keys:
        matched |= secrets.compare_digest(provided, key.encode("utf-8"))
    if not provided or not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
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
    dependencies=[Depends(require_api_key)],
    tags=["classify"],
)
async def classify(
    payload: ClassifyRequest,
    settings: Settings = Depends(get_settings),
) -> ClassifyResponse:
    _ensure_loaded()
    template = payload.template(settings.default_hypothesis_template)

    async with _semaphore():
        prediction = await run_in_threadpool(
            classifier.classify,
            payload.text,
            payload.labels,
            payload.multi_label,
            template,
        )

    response = _to_response(prediction, payload.multi_label, settings)
    _log_result("classify", response, len(payload.labels))
    return response


@app.post(
    "/classify/batch",
    response_model=BatchResponse,
    dependencies=[Depends(require_api_key)],
    tags=["classify"],
)
async def classify_batch(
    payload: BatchRequest,
    settings: Settings = Depends(get_settings),
) -> BatchResponse:
    _ensure_loaded()
    items = [
        {
            "text": item.text,
            "labels": item.labels,
            "multi_label": item.multi_label,
            "hypothesis_template": item.template(settings.default_hypothesis_template),
        }
        for item in payload.items
    ]

    async with _semaphore():
        predictions: List[Prediction] = await run_in_threadpool(
            classifier.classify_batch, items
        )

    responses = [
        _to_response(prediction, item.multi_label, settings)
        for prediction, item in zip(predictions, payload.items)
    ]
    for response, item in zip(responses, payload.items):
        _log_result("classify_batch", response, len(item.labels))
    return BatchResponse(results=responses)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _semaphore() -> asyncio.Semaphore:
    if _inference_semaphore is None:  # pragma: no cover - only outside the lifespan
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service is still starting",
        )
    return _inference_semaphore


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


def _log_result(event: str, response: ClassifyResponse, n_labels: int) -> None:
    """Logs metrics only - never the text itself."""
    logger.info(
        event,
        extra={
            "duration_ms": response.duration_ms,
            "n_labels": n_labels,
            "top_label": response.label,
            "score": response.score,
            "multi_label": response.multi_label,
        },
    )
