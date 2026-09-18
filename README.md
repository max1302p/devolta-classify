# classify

A small self-hosted HTTP service for zero-shot text classification on CPU.
You send text and a list of labels, it returns the labels with confidence scores.
No GPU, no external API, no per-request model loading.

```bash
curl -X POST https://classify.example.com/classify \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
        "text": "Mein Kassensystem druckt seit gestern keine Bons mehr",
        "labels": ["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"]
      }'
```

```json
{
  "label": "Technisches Problem",
  "score": 0.7384,
  "results": [
    {"label": "Technisches Problem", "score": 0.7384},
    {"label": "Rechnung", "score": 0.1891},
    {"label": "Feature-Wunsch", "score": 0.0591},
    {"label": "Spam", "score": 0.0133}
  ],
  "multi_label": false,
  "model": "MoritzLaurer/bge-m3-zeroshot-v2.0",
  "duration_ms": 478
}
```

The service was built for German-language text, which is why the default
hypothesis template is German (`Diese Nachricht betrifft {}.`). The model is
multilingual and English works too — override the template per request.

## What it is good at, and what it is not

This matters more than the usual feature list, so it comes first. Measured over
144 cases across 15 domains (see [docs/benchmark.md](docs/benchmark.md)):

Where the label is simply **the topic of the text**, accuracy is 1.000 — news
sections, email triage, customer intents, job ad industries, review aspects,
English support tickets.

Where the label requires a **judgement that is not in the text**, it drops
sharply:

| Domain | Accuracy | What the model would have to do |
|---|---|---|
| medical_triage | 0.375 | decide whether symptoms need a doctor |
| formality | 0.625 | judge register rather than recognise content |
| cuisine | 0.625 | recall world knowledge about dishes |
| urgency | 0.750 | reason about consequences |

The errors are systematic, not scattered: in `medical_triage` all three misses
go the same direction, in `formality` all three do as well. That is a limit of
the method, not of this particular model. NLI zero-shot checks whether a
sentence follows from a text; it does not check whether a conclusion is
professionally correct. For those tasks you want a fine-tuned model or an LLM.

Overall: **0.868** with the default template, **0.917** with a per-domain
template.

## Model

Default is [`MoritzLaurer/bge-m3-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0).
The obvious alternative for multilingual zero-shot is mDeBERTa; it lost clearly
on both axes:

| Model | Params | Accuracy (default template) | Accuracy (best template) | p50 |
|---|---|---|---|---|
| **bge-m3-zeroshot-v2.0** | 568M | **0.868** | **0.917** | 1.9 s |
| mDeBERTa-v3-base-xnli-…-2mil7 | 279M | 0.722 | 0.792 | 2.3 s |
| multilingual-MiniLMv2-L6-mnli-xnli | 107M | 0.486 | 0.521 | 0.14 s |

<sub>PyTorch backend, 4 threads. The ONNX numbers are below.</sub>

bge-m3 is faster despite having twice the parameters: DeBERTa's disentangled
attention costs more on CPU than the extra layers do. MiniLM is 14x faster but
barely beats guessing at four labels (0.486 against a 0.25 baseline).

Swap models with `MODEL_ID` — as an environment variable **and** as a build
argument, since the model has to be inside the image:

```bash
docker build --build-arg MODEL_ID=some-org/some-model -t classify .
```

### A note on the model licence

The model is MIT licensed, and this repository ships no weights, so publishing
and redistributing the code is unaffected. For commercial use it is worth
knowing that the `zeroshot-v2.0` variants **without** a `-c` suffix were trained
on a mix of datasets with mixed licences, while the `-c` variants use only
commercially-friendly data. If that matters for your deployment, evaluate
`MoritzLaurer/bge-m3-zeroshot-v2.0-c` — `scripts/benchmark.py` exists for
exactly that kind of comparison.

## Backend: ONNX Runtime

Inference runs through ONNX Runtime rather than PyTorch. Same weights, same
arithmetic, different execution graph:

| | Accuracy | p50 | Load time | RSS |
|---|---|---|---|---|
| **ONNX Runtime** (default) | **0.868** | **339 ms** | 5.9 s | 1.9 GB |
| PyTorch | 0.868 | 1911 ms | 1.6 s | 0.5 GB |

**5.6x faster, and not one of the 144 predictions changed.** fp32 ONNX computes
the same thing as fp32 PyTorch; the gain comes from operator fusion and a
leaner runtime stack. Optimising ahead of time is why loading takes longer — a
trade you want, since the model loads once at startup.

int8 quantisation is deliberately **not** enabled. Two measurements argued
against it: the quantisation step itself needs well over 6 GB of RAM (a risky
build step on a shared host), and PyTorch's dynamic int8 was *slower* than fp32
(3484 ms against ~1900 ms) because ARM CPUs lack the matching integer
instructions. `--build-arg ONNX_QUANTIZE=1` is still there if you want to
measure it on your own hardware.

`BACKEND=torch` switches back, but the image then has to be built with
`--build-arg BACKEND=torch`, otherwise the Torch weights are missing.

## Quickstart

```bash
cp .env.example .env          # set API_KEYS
docker compose up --build     # first build downloads the model (~2.3 GB)
curl -s localhost:8000/health
```

The build bakes the model into the image and `HF_HUB_OFFLINE=1` enforces that
nothing is fetched at runtime. The image is around 3.5 GB as a result, and the
container starts without network access.

## API

All endpoints except `/health` require `Authorization: Bearer <key>` (or
`X-API-Key: <key>`). Errors always come back as `{"error": "..."}`.
OpenAPI docs are at `/docs`.

### `POST /classify`

| Field | Required | Notes |
|---|---|---|
| `text` | yes | 1–5000 characters |
| `labels` | yes | 2–50 unique, non-empty strings |
| `multi_label` | no | default `false`; `true` scores each label independently |
| `hypothesis_template` | no | must contain `{}`; defaults to `DEFAULT_HYPOTHESIS_TEMPLATE` |

`results` is sorted by score descending, scores are rounded to 4 decimals, and
`label`/`score` mirror the top result.

### `POST /classify/batch`

Takes `{"items": [ ... ]}` with up to 32 items and returns
`{"results": [ ... ]}` in the same order. Cheaper per text than individual
calls, since the request overhead is paid once.

### `GET /health`

`200 {"status":"ok","model_loaded":true}` once the model is loaded,
`503 {"status":"loading","model_loaded":false}` before that. No auth.

## Configuration

Everything is configured through environment variables.

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | — (**required**) | Comma separated API keys. `API_KEY` also works for a single key |
| `NUM_THREADS` | CPU count | Threads for inference |
| `MAX_CONCURRENCY` | `2` | Concurrent inferences (semaphore) |
| `DEFAULT_HYPOTHESIS_TEMPLATE` | `Diese Nachricht betrifft {}.` | Must contain `{}` |
| `PORT` | `8000` | Uvicorn port |
| `MODEL_ID` | `MoritzLaurer/bge-m3-zeroshot-v2.0` | Must match what the image was built with |
| `BACKEND` | `onnx` | `onnx` or `torch`, must match the build |
| `ONNX_DIR` | `/opt/onnx` | ONNX export inside the image |

Missing keys or an invalid `BACKEND` abort startup rather than failing later.

Key rotation without downtime: append the new key to `API_KEYS`, move clients
over, then drop the old one.

## Running on a shared host

The service is built to stay inside its lane:

- `deploy.resources.limits.cpus` in the compose file is a hard cgroup ceiling —
  the container cannot take more CPU no matter how many requests arrive.
- `NUM_THREADS` and `OMP_NUM_THREADS` cap the Torch and ONNX thread pools, and
  inter-op parallelism is pinned to 1 so the pools cannot oversubscribe the CPU.
- `MAX_CONCURRENCY` bounds concurrent inferences with a semaphore; further
  requests queue instead of fighting over cores. With the default of 2, four
  parallel requests come back in two waves — and `/health` still answers in
  ~110 ms, because the blocking work happens in a thread pool and the event
  loop stays free.
- Memory limit 5 GB. Measured with ONNX: ~1.9 GB steady state, ~3.2 GB briefly
  while loading. ONNX Runtime keeps the weights in process memory; PyTorch would
  map them into the page cache (0.5 GB there).

Rule of thumb: `NUM_THREADS × MAX_CONCURRENCY ≲ CPU_LIMIT × 2`.

### Behind a reverse proxy

Point the proxy at port 8000 and remove the `ports:` block from the compose
file so the port is not published on the host. Uvicorn runs with
`--proxy-headers`. The health check has a 180 s start period to cover model
loading.

## Development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install -r requirements-dev.txt

export API_KEYS=dev-key BACKEND=torch
uvicorn app.main:app --reload
```

### Tests

```bash
pytest                      # fast tests, no model (~1 s)
pytest -m slow              # integration test against the real model
docker build --target test  # both, inside the image
```

The fast tests patch the pipeline away and cover validation, auth, ordering,
rounding, batch order, and that request text never reaches the logs.

### Benchmarking your own labels

`scripts/benchmark_data.json` is the template: your domains, your labels, your
templates, 10–20 cases each.

```bash
# against a model, locally
docker run --rm -e NUM_THREADS=4 -e HF_HOME=/opt/hf \
  -e BENCHMARK_DATA=/bench/my_data.json \
  -v "$PWD/scripts:/bench:ro" \
  --entrypoint python classify:latest /bench/benchmark.py

# against a running instance - measures what is actually deployed
BASE=https://classify.example.com KEY=... python scripts/benchmark_http.py
```

Two things worth knowing before you tune anything:

**The default template is a decent default, not an optimum.** 0.868 against
0.917 for the best per-domain template. The gap comes almost entirely from
domains where the default reads as broken German ("Diese Nachricht betrifft
förmlich.").

**Rewriting labels helps, but not reliably.** Descriptive instead of terse
labels improved three domains by 25 points and hurt two by 13–20 points; net
+0.07 over 42 cases, which is three cases and not conclusive. It is a real
lever, but one to measure rather than trust.

## Layout

```
app/main.py            FastAPI app, routes, lifespan, auth, error format
app/classifier.py      model wrapper (load, classify, classify_batch)
app/schemas.py         Pydantic v2 models
app/config.py          environment configuration
app/logging_config.py  JSON logging
tests/                 fast tests plus a slow integration test
scripts/               model download, ONNX export, smoke test, benchmarks
docs/benchmark.md      measurements: 3 models, 15 domains, 144 cases
```

Logging is structured JSON on stdout with `duration_ms`, `n_labels`,
`top_label`, `score` and `multi_label`. **The text itself is never logged**, and
there is a test that keeps it that way. Nothing is persisted.

## Licence

MIT, see [LICENSE](LICENSE). The model is MIT as well; see the note above if you
have strict requirements about training data provenance.
