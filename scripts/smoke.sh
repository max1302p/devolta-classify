#!/usr/bin/env bash
# Smoke test against a running instance:
#   BASE=https://classify.example.com KEY=... ./scripts/smoke.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
KEY="${KEY:?KEY (API key) must be set}"
RUNS="${RUNS:-5}"

PAYLOAD='{
  "text": "Mein Kassensystem druckt seit gestern keine Bons mehr",
  "labels": ["Technisches Problem", "Rechnung", "Feature-Wunsch", "Spam"],
  "multi_label": false,
  "hypothesis_template": "Diese Nachricht betrifft {}."
}'

echo "== GET $BASE/health"
curl -fsS -w '\nhttp=%{http_code} time=%{time_total}s\n' "$BASE/health"

echo
echo "== POST $BASE/classify"
curl -fsS -X POST "$BASE/classify" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  -w '\nhttp=%{http_code} time=%{time_total}s\n'

echo
echo "== latency over $RUNS requests (seconds, end to end)"
for _ in $(seq "$RUNS"); do
  curl -fsS -o /dev/null -X POST "$BASE/classify" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    -w '%{time_total}\n'
done

echo
echo "== auth must fail without a key (expecting 401)"
curl -s -o /dev/null -w 'http=%{http_code}\n' -X POST "$BASE/classify" \
  -H "Content-Type: application/json" -d "$PAYLOAD"
