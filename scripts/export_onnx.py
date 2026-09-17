"""Export the model to ONNX and optionally quantise it to int8.

Runs during the Docker build. The result (graph + tokenizer + config) lands in
ONNX_DIR and loads at runtime without the HF cache and without network access.

Env:
  MODEL_ID        source model on the Hub
  ONNX_DIR        target directory (default /opt/onnx)
  ONNX_QUANTIZE   "0" (default) keeps fp32, "1" quantises to int8 dynamically
                  (int8 needs well over 6 GB of RAM during the build)
"""

from __future__ import annotations

import os
import pathlib
import platform
import shutil
import sys

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

QUANTIZED_NAME = "model_quantized.onnx"


def _cpu_flags() -> set[str]:
    try:
        text = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags") or line.startswith("Features"):
            return set(line.split(":", 1)[1].split())
    return set()


def _quantization_config():
    """Pick the config matching the build CPU - builds happen on the target host."""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        print("quantization: arm64", flush=True)
        return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

    flags = _cpu_flags()
    if "avx512_vnni" in flags:
        print("quantization: avx512_vnni", flush=True)
        return AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    if "avx512f" in flags:
        print("quantization: avx512", flush=True)
        return AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
    print("quantization: avx2 (reduce_range)", flush=True)
    return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)


def _load_fp32(model_id: str):
    """Reuse the repo's ONNX export if there is one, otherwise export ourselves.

    Exporting traces the Torch model and needs several times the model size in
    RAM; a ready-made export in the model repo avoids that entirely.
    """
    try:
        print(f"loading prebuilt ONNX export from {model_id} (subfolder onnx)", flush=True)
        return ORTModelForSequenceClassification.from_pretrained(model_id, subfolder="onnx")
    except Exception as exc:  # noqa: BLE001 - model repo has no onnx folder
        print(f"  no prebuilt export ({type(exc).__name__}), exporting instead", flush=True)
        return ORTModelForSequenceClassification.from_pretrained(model_id, export=True)


def _dir_size_mb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    model_id = os.environ.get("MODEL_ID") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not model_id:
        print("MODEL_ID is missing", file=sys.stderr)
        return 1

    out = pathlib.Path(os.environ.get("ONNX_DIR", "/opt/onnx"))
    out.mkdir(parents=True, exist_ok=True)

    model = _load_fp32(model_id)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    print(f"fp32 export: {_dir_size_mb(out):.0f} MB", flush=True)

    if os.environ.get("ONNX_QUANTIZE", "0") != "1":
        _smoke_test(out)
        return 0

    quantizer = ORTQuantizer.from_pretrained(out)
    quantizer.quantize(save_dir=out, quantization_config=_quantization_config())

    # Drop the fp32 weights, otherwise the image carries both variants.
    if (out / QUANTIZED_NAME).exists():
        for leftover in ("model.onnx", "model.onnx_data"):
            target = out / leftover
            if target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
    print(f"int8 done: {_dir_size_mb(out):.0f} MB", flush=True)
    _smoke_test(out)
    return 0


def _smoke_test(path: pathlib.Path) -> None:
    """Make sure the exported model loads and returns something sensible."""
    from transformers import pipeline

    kwargs = {"file_name": QUANTIZED_NAME} if (path / QUANTIZED_NAME).is_file() else {}
    model = ORTModelForSequenceClassification.from_pretrained(str(path), **kwargs)
    pipe = pipeline("zero-shot-classification", model=model,
                    tokenizer=AutoTokenizer.from_pretrained(str(path)))
    result = pipe(
        "Mein Kassensystem druckt seit gestern keine Bons mehr",
        candidate_labels=["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"],
        hypothesis_template="Diese Nachricht betrifft {}.",
    )
    print(f"smoke test: {result['labels'][0]} {result['scores'][0]:.4f}", flush=True)
    if result["labels"][0] != "Technisches Problem":
        raise SystemExit("ONNX smoke test failed: unexpected top label")


if __name__ == "__main__":
    raise SystemExit(main())
