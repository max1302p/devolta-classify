# Benchmark: models, domains and hypothesis templates

Reproducible with `scripts/benchmark.py` over `scripts/benchmark_data.json`:
**15 domains, 144 cases, 2-6 labels per domain, 2-3 templates per domain**
(412 classifications per model). Measured inside the runtime image, 4 CPU
threads, arm64 container, `multi_label=false`.

`global` is the service default template `Diese Nachricht betrifft {}.`,
`best` is the best of the domain-specific templates.

The dataset is deliberately broad: support tickets, sentiment, news sections,
email triage, customer intents, urgency, comment moderation, register, job
ads, document types, review aspects, English text, cuisines and symptom
triage. A model that only does well on one of those is not much use.

## Models

| Model | Params | Accuracy (default template) | Accuracy (best template) | p50 |
|---|---|---|---|---|
| **bge-m3-zeroshot-v2.0** | 568M | 0.868 | 0.917 | 1911 ms |
| mDeBERTa-v3-base-xnli | 279M | 0.722 | 0.792 | 2331 ms |
| MiniLMv2-L6 | 107M | 0.486 | 0.521 | 138 ms |

bge-m3 wins 13 of 15 domains. It is faster than mDeBERTa despite twice the
parameter count, because DeBERTa's disentangled attention costs more on CPU
than the additional layers do. MiniLM is 14x faster and barely above chance
(0.486 with a 0.25 baseline at four labels).

## Accuracy per domain

| Domain | Labels | bge-m3 global | bge-m3 best | mDeBERTa global | MiniLM global |
|---|---|---|---|---|---|
| support_ticket | 4 | 0.917 | 0.917 | 0.417 | 0.500 |
| sentiment | 3 | 0.900 | 1.000 | 0.600 | 0.600 |
| news_section | 6 | 1.000 | 1.000 | 1.000 | 0.333 |
| email_triage | 5 | 1.000 | 1.000 | 0.800 | 0.600 |
| intent_noun_labels | 5 | 1.000 | 1.000 | 1.000 | 0.500 |
| intent_verb_labels | 5 | 0.900 | 1.000 | 1.000 | 0.400 |
| urgency | 2 | 0.750 | 0.875 | 0.625 | 0.500 |
| moderation | 4 | 0.800 | 0.900 | 0.600 | 0.300 |
| formality | 2 | 0.625 | 0.625 | 0.375 | 0.500 |
| job_ads | 5 | 1.000 | 1.000 | 0.800 | 0.600 |
| document_type | 5 | 0.900 | 0.900 | 0.900 | 0.500 |
| review_aspect | 4 | 1.000 | 1.000 | 0.700 | 1.000 |
| english_tickets | 4 | 1.000 | 1.000 | 0.375 | 0.250 |
| cuisine | 5 | 0.625 | 0.750 | 1.000 | 0.250 |
| medical_triage | 3 | 0.375 | 0.625 | 0.500 | 0.375 |

## Topic assignment works, judgement calls do not

Wherever the label is simply the topic of the text, bge-m3 scores 1.000:
news sections, email triage, customer intents, job ad industries, review
aspects, English tickets. It breaks down wherever the label requires a
judgement that is not stated in the text:

| Domain | Accuracy | What the model would have to do |
|---|---|---|
| medical_triage | 0.375 | decide whether symptoms warrant a doctor |
| formality | 0.625 | judge register instead of recognising content |
| cuisine | 0.625 | recall world knowledge about dishes |
| urgency | 0.750 | reason about consequences ("server down" means work stops) |

The failures are not scattered. In `medical_triage` all three misses go
`Arzttermin` -> `Selbstbehandlung`; in `formality` all three go
`förmlich` -> `umgangssprachlich`. That is a property of the method: NLI
zero-shot asks whether a hypothesis follows from a text, not whether a
conclusion is professionally sound. Those tasks need a fine-tuned model or an
LLM, not a different template.

## Templates and label wording

Over all 144 cases: 0.868 with the default template against 0.917 with the best
per-domain template. The gain comes almost entirely from domains where the
default template reads as broken German ("Diese Nachricht betrifft förmlich.").
Anything running in production with a fixed label set is worth measuring two or
three templates for once.

Descriptive instead of terse labels, tested separately
(`scripts/benchmark_data_labels.json`, 42 cases, bge-m3):

| Domain | terse labels | descriptive labels | delta |
|---|---|---|---|
| formality | 0.500 | 0.750 | +0.25 |
| urgency | 0.750 | 1.000 | +0.25 |
| cuisine | 0.750 | 1.000 | +0.25 |
| medical_triage | 0.625 | 0.500 | -0.13 |
| moderation | 0.900 | 0.700 | -0.20 |
| **total** | **0.714** | **0.786** | **+0.07** |

Three clear wins, two clear losses. The net gain of +0.07 is three cases and
not meaningful at this sample size. Label wording is a real lever, but one to
measure for your own labels rather than to trust.

## Measuring your own labels

Copy `scripts/benchmark_data.json`, replace the domains with your own labels,
templates and 10-20 cases each, then:

```bash
docker run --rm -e NUM_THREADS=4 -e HF_HOME=/opt/hf \
  -e BENCHMARK_DATA=/bench/my_data.json \
  -v "$PWD/scripts:/bench:ro" \
  --entrypoint python classify:latest /bench/benchmark.py
```

Raw per-case results for every run above are in `docs/bench-*.json`.
