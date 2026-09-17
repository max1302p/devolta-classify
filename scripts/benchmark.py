"""Zero-shot benchmark across models, domains and hypothesis templates.

    python scripts/benchmark.py MODEL [MODEL ...]

Dataset: scripts/benchmark_data.json (15 domains, 144 cases).
Output: JSON lines on stdout plus a summary; raw records via RESULTS_OUT.

A model argument is a Hub id, "quant:<hub id>" for PyTorch dynamic int8, or
"onnx:<path>" for an ONNX export produced by scripts/export_onnx.py.
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

import torch
from transformers import pipeline

DATA_PATH = pathlib.Path(os.environ.get("BENCHMARK_DATA", pathlib.Path(__file__).with_name("benchmark_data.json")))
RESULTS_OUT = os.environ.get("RESULTS_OUT")
DEFAULT_MODELS = ["MoritzLaurer/bge-m3-zeroshot-v2.0"]


def build_pipeline(spec: str, num_threads: int):
    """Build a zero-shot pipeline for one model spec."""
    if spec.startswith("quant:"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_id = spec.split(":", 1)[1]
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return pipeline("zero-shot-classification", model=quantized,
                        tokenizer=tokenizer, device=-1), f"torch-int8-{model_id.split('/')[-1]}"

    if not spec.startswith("onnx:"):
        return pipeline("zero-shot-classification", model=spec, tokenizer=spec, device=-1), spec

    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    path = spec.split(":", 1)[1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    kwargs = {}
    if any(f.name == "model_quantized.onnx" for f in pathlib.Path(path).iterdir()):
        kwargs["file_name"] = "model_quantized.onnx"
    model = ORTModelForSequenceClassification.from_pretrained(
        path, session_options=options, provider="CPUExecutionProvider", **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(path)
    name = f"onnx-{'int8' if kwargs else 'fp32'}"
    return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer), name


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def main() -> int:
    torch.set_num_threads(int(os.environ.get("NUM_THREADS", "4")))
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    domains = data["domains"]
    models = sys.argv[1:] or DEFAULT_MODELS

    records: list[dict] = []
    summary: list[dict] = []

    threads = int(os.environ.get("NUM_THREADS", "4"))
    for model_id in models:
        t0 = time.perf_counter()
        pipe, resolved = build_pipeline(model_id, threads)
        short = resolved.split("/")[-1]
        print(json.dumps({"model": short, "load_s": round(time.perf_counter() - t0, 1)}), flush=True)

        for domain in domains:
            labels = domain["labels"]
            for tname, template in domain["templates"].items():
                correct, margins, times = 0, [], []
                with torch.inference_mode():
                    for case in domain["cases"]:
                        started = time.perf_counter()
                        out = pipe(case["text"], candidate_labels=labels, hypothesis_template=template)
                        times.append((time.perf_counter() - started) * 1000)
                        pred, score = out["labels"][0], out["scores"][0]
                        hit = pred == case["gold"]
                        correct += hit
                        margins.append(score - out["scores"][1])
                        records.append({
                            "model": short, "domain": domain["name"], "template": tname,
                            "text": case["text"], "gold": case["gold"], "pred": pred,
                            "score": round(score, 4), "hit": hit,
                        })
                row = {
                    "model": short, "domain": domain["name"], "template": tname,
                    "n": len(domain["cases"]), "n_labels": len(labels),
                    "acc": round(correct / len(domain["cases"]), 3),
                    "avg_margin": round(sum(margins) / len(margins), 3),
                    "p50_ms": round(percentile(times, 0.5)),
                    "p95_ms": round(percentile(times, 0.95)),
                }
                summary.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

        del pipe
        gc.collect()

    if RESULTS_OUT:
        pathlib.Path(RESULTS_OUT).write_text(
            json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    _report(summary, records, domains)
    return 0


def _report(summary: list[dict], records: list[dict], domains: list[dict]) -> None:
    models = sorted({row["model"] for row in summary})

    print("\n=== accuracy per domain x model/template ===", flush=True)
    for domain in domains:
        print(f"\n{domain['name']}  ({len(domain['cases'])} cases, {len(domain['labels'])} labels)")
        for row in summary:
            if row["domain"] == domain["name"]:
                print(f"   {row['acc']:.3f}  margin={row['avg_margin']:.2f}  "
                      f"p50={row['p50_ms']:>5}ms  {row['model']:<28} {row['template']}")

    print("\n=== totals per model ===", flush=True)
    for model in models:
        rows = [r for r in summary if r["model"] == model]
        glob = [r for r in rows if r["template"] == "global"]
        best = defaultdict(float)
        for row in rows:
            best[row["domain"]] = max(best[row["domain"]], row["acc"])

        cases_glob = [r for r in records if r["model"] == model and r["template"] == "global"]
        micro_glob = sum(r["hit"] for r in cases_glob) / max(len(cases_glob), 1)
        times = [r["p50_ms"] for r in rows]
        print(f"\n{model}")
        print(f"  default template, micro over all cases   : {micro_glob:.3f}")
        print(f"  default template, macro over domains     : {sum(r['acc'] for r in glob) / len(glob):.3f}")
        print(f"  best template per domain, macro          : {sum(best.values()) / len(best):.3f}")
        print(f"  p50 latency across domains: min {min(times)} ms, median {percentile([float(t) for t in times], 0.5):.0f} ms, max {max(times)} ms")

    print("\n=== errors with the default template ===", flush=True)
    errors = defaultdict(list)
    for rec in records:
        if rec["template"] == "global" and not rec["hit"]:
            errors[rec["model"]].append(rec)
    for model in models:
        print(f"\n{model}: {len(errors[model])} errors")
        for rec in errors[model][:25]:
            print(f"   [{rec['domain']}] gold={rec['gold']!r} pred={rec['pred']!r} ({rec['score']}) :: {rec['text'][:60]}")


if __name__ == "__main__":
    raise SystemExit(main())
