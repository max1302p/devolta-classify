"""Run the benchmark against a live instance - measures what is deployed.

    BASE=https://classify.example.com KEY=... python scripts/benchmark_http.py

Env:
  BASE            base URL (default http://localhost:8000)
  KEY             API key (required)
  BENCHMARK_DATA  dataset (default scripts/benchmark_data.json)
  TEMPLATES       "global" (default) or "all"
  BATCH           items per /classify/batch request (default 16)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

BASE = os.environ.get("BASE", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("KEY", "")
DATA_PATH = pathlib.Path(os.environ.get("BENCHMARK_DATA", pathlib.Path(__file__).with_name("benchmark_data.json")))
ONLY_GLOBAL = os.environ.get("TEMPLATES", "global") != "all"
BATCH = int(os.environ.get("BATCH", "16"))


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> int:
    if not KEY:
        print("KEY is missing", file=sys.stderr)
        return 1

    health = json.loads(urllib.request.urlopen(f"{BASE}/health", timeout=30).read())
    print(f"health: {health}", flush=True)

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    items, meta = [], []
    for domain in data["domains"]:
        for tname, template in domain["templates"].items():
            if ONLY_GLOBAL and tname != "global":
                continue
            for case in domain["cases"]:
                items.append({"text": case["text"], "labels": domain["labels"],
                              "hypothesis_template": template})
                meta.append((domain["name"], tname, case["gold"]))

    print(f"{len(items)} cases against {BASE} ...", flush=True)
    results, wall = [], time.perf_counter()
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        try:
            results.extend(post("/classify/batch", {"items": chunk})["results"])
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode()[:200]}", file=sys.stderr)
            return 1
        print(f"  {len(results)}/{len(items)}", flush=True)
    wall = time.perf_counter() - wall

    hits = defaultdict(list)
    durations = []
    errors = []
    for (domain, template, gold), result in zip(meta, results):
        ok = result["label"] == gold
        hits[(domain, template)].append(ok)
        durations.append(result["duration_ms"])
        if not ok:
            errors.append((domain, gold, result["label"], result["score"]))

    print("\n=== accuracy per domain ===")
    for (domain, template), values in hits.items():
        print(f"  {sum(values) / len(values):.3f}  {domain} ({template})")

    total = [v for values in hits.values() for v in values]
    durations.sort()
    print(f"\nmicro over {len(total)} cases: {sum(total) / len(total):.3f}")
    print(f"duration_ms: p50={durations[len(durations)//2]}  "
          f"p95={durations[int(len(durations)*0.95)]}  "
          f"min={durations[0]}  max={durations[-1]}")
    print(f"wall clock: {wall:.1f}s for {len(items)} classifications "
          f"({wall / len(items) * 1000:.0f} ms per case including batching)")

    print(f"\n=== {len(errors)} errors ===")
    for domain, gold, pred, score in errors[:30]:
        print(f"  [{domain}] gold={gold!r} pred={pred!r} ({score})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
