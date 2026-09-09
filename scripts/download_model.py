"""Download the model into HF_HOME during the image build and smoke-test it."""

from __future__ import annotations

import os
import sys

from transformers import pipeline


def main() -> int:
    model_id = os.environ.get("MODEL_ID") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not model_id:
        print("MODEL_ID is missing", file=sys.stderr)
        return 1

    print(f"downloading {model_id} into {os.environ.get('HF_HOME')}", flush=True)
    pipe = pipeline(
        "zero-shot-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1,
    )
    result = pipe(
        "Mein Kassensystem druckt seit gestern keine Bons mehr",
        candidate_labels=["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"],
        hypothesis_template="Diese Nachricht betrifft {}.",
    )
    print(f"smoke test: {result['labels'][0]} {result['scores'][0]:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
