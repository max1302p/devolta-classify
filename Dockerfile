# syntax=docker/dockerfile:1
#
# Multi-stage build: CPU-only Torch plus the model baked into the image, so the
# container starts without network access and without a cold download.

ARG PYTHON_IMAGE=python:3.11-slim
ARG MODEL_ID=MoritzLaurer/bge-m3-zeroshot-v2.0
ARG TORCH_VERSION=2.8.0
# onnx = ONNX Runtime (roughly 5x faster on CPU), torch = PyTorch weights
ARG BACKEND=onnx
# int8 needs well over 6 GB of RAM during the build and buys nothing measurable
# over fp32 ONNX - see docs/benchmark.md. Off by default.
ARG ONNX_QUANTIZE=0

# --------------------------------------------------------------------------- #
# 1. Builder: virtualenv and model artifacts
# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS builder

ARG MODEL_ID
ARG TORCH_VERSION
ARG BACKEND
ARG ONNX_QUANTIZE

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/opt/hf

WORKDIR /build

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements.txt ./
# CPU wheels first, otherwise pip pulls the CUDA build from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install -r requirements.txt

# Pull the model into the image at build time, as an ONNX graph or as Torch
# weights. Both paths smoke-test the model, so a broken build fails right here.
COPY scripts/download_model.py scripts/export_onnx.py ./scripts/
ENV ONNX_DIR=/opt/onnx
RUN mkdir -p /opt/onnx /opt/hf \
    && if [ "${BACKEND}" = "onnx" ]; then \
         MODEL_ID="${MODEL_ID}" ONNX_QUANTIZE="${ONNX_QUANTIZE}" python scripts/export_onnx.py \
         && rm -rf /opt/hf/hub; \
       else \
         MODEL_ID="${MODEL_ID}" python scripts/download_model.py; \
       fi

# --------------------------------------------------------------------------- #
# 2. Test stage (optional): docker build --target test .
# --------------------------------------------------------------------------- #
FROM builder AS test

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

WORKDIR /app
COPY app ./app
COPY tests ./tests
COPY pytest.ini ./

ENV API_KEYS=test-key,zweiter-key \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN python -m pytest

# --------------------------------------------------------------------------- #
# 3. Runtime: slim, non-root, offline
# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS runtime

ARG MODEL_ID
ARG BACKEND

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_ID=${MODEL_ID} \
    BACKEND=${BACKEND} \
    ONNX_DIR=/opt/onnx \
    PORT=8000

RUN groupadd --system app \
    && useradd --system --gid app --uid 10001 --create-home --home-dir /home/app app

COPY --from=builder --chown=root:root /opt/venv /opt/venv
COPY --from=builder --chown=root:root /opt/hf /opt/hf
COPY --from=builder --chown=root:root /opt/onnx /opt/onnx

WORKDIR /app
COPY --chown=root:root app ./app

USER app
EXPOSE 8000

# Generous start period: loading the model takes a few seconds on CPU.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=5).read()"]

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
