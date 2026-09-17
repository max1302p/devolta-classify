"""Model wrapper around the zero-shot pipeline (CPU only)."""

from __future__ import annotations

import contextlib
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Iterable, List, Sequence, Tuple

from .config import MODEL_ID, Settings

logger = logging.getLogger("classify.classifier")

SCORE_DIGITS = 4
QUANTIZED_MODEL = "model_quantized.onnx"


@dataclass(frozen=True)
class Prediction:
    """Result of a single classification."""

    results: List[Tuple[str, float]]
    duration_ms: int

    @property
    def top(self) -> Tuple[str, float]:
        return self.results[0]


class ModelNotLoadedError(RuntimeError):
    pass


class Classifier:
    """Loads the model once and classifies text synchronously on the CPU."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._pipe: Callable[..., dict] | None = None
        self._inference_ctx: Callable[[], ContextManager] = contextlib.nullcontext

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self, settings: Settings) -> None:
        """Load the model once, during the FastAPI lifespan."""
        if self.is_loaded:
            return

        import torch  # imported lazily so the unit tests do not need torch

        self.model_id = settings.model_id or self.model_id
        torch.set_num_threads(settings.num_threads)
        try:
            # A single inter-op thread keeps the thread pools from
            # oversubscribing the CPU and starving everything else on the host.
            torch.set_num_interop_threads(1)
        except RuntimeError:  # already initialised, not critical
            pass
        self._inference_ctx = torch.inference_mode
        started = time.perf_counter()
        self._pipe = self._build_pipeline(settings)
        logger.info(
            "model_loaded",
            extra={
                "model": self.model_id,
                "backend": settings.backend,
                "num_threads": settings.num_threads,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    def _build_pipeline(self, settings: Settings) -> Callable[..., dict]:
        from transformers import pipeline

        if settings.backend == "onnx":
            model, tokenizer = self._load_onnx(settings)
            return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

        return pipeline(
            "zero-shot-classification",
            model=self.model_id,
            tokenizer=self.model_id,
            device=-1,  # CPU
        )

    def _load_onnx(self, settings: Settings):
        """Load the ONNX graph from ONNX_DIR, produced during the image build."""
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        path = pathlib.Path(settings.onnx_dir)
        if not path.is_dir():
            raise FileNotFoundError(
                f"ONNX_DIR {path} does not exist - was the image built without the "
                "ONNX export? Set BACKEND=torch instead."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        kwargs = {}
        if (path / QUANTIZED_MODEL).is_file():
            kwargs["file_name"] = QUANTIZED_MODEL

        model = ORTModelForSequenceClassification.from_pretrained(
            str(path),
            session_options=options,
            provider="CPUExecutionProvider",
            **kwargs,
        )
        return model, AutoTokenizer.from_pretrained(str(path))

    def classify(
        self,
        text: str,
        labels: Sequence[str],
        multi_label: bool,
        hypothesis_template: str,
    ) -> Prediction:
        if self._pipe is None:
            raise ModelNotLoadedError("model is not loaded")

        started = time.perf_counter()
        with self._inference_ctx():
            raw = self._pipe(
                text,
                candidate_labels=list(labels),
                multi_label=multi_label,
                hypothesis_template=hypothesis_template,
            )
        duration_ms = int(round((time.perf_counter() - started) * 1000))
        return Prediction(results=_to_sorted_results(raw), duration_ms=duration_ms)

    def classify_batch(self, items: Iterable[dict]) -> List[Prediction]:
        """Classify items sequentially inside the same worker thread."""
        return [
            self.classify(
                text=item["text"],
                labels=item["labels"],
                multi_label=item["multi_label"],
                hypothesis_template=item["hypothesis_template"],
            )
            for item in items
        ]


def _to_sorted_results(raw: dict) -> List[Tuple[str, float]]:
    """Turn the pipeline output into rounded pairs, highest score first."""
    if isinstance(raw, list):  # defensive: the pipeline can return lists
        raw = raw[0]
    pairs = list(zip(raw["labels"], raw["scores"]))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [(label, round(float(score), SCORE_DIGITS)) for label, score in pairs]


classifier = Classifier()
