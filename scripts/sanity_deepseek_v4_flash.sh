#!/usr/bin/env bash
# Qualitative sanity-check the served DSv4-Flash output on a small fixed set
# of deterministic (T=0, greedy) prompts with known ground truth. Reads
# completions through the server — same path as the eval harness — and
# prints them next to the expected answers so you can eyeball whether
# the model is producing coherent text vs garbage.
#
# Usage:
#   scripts/sanity_deepseek_v4_flash.sh
#   SERVED_NAME=foo HOST=… PORT=… scripts/sanity_deepseek_v4_flash.sh

set -uo pipefail

SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"

# Each test: prompt → expected substring(s) (case-insensitive `grep -i`)
# Greedy (T=0) so output is deterministic and reproducible across runs.
tests=(
  "The capital of France is|paris"
  "2 + 2 =|4"
  "Q: What is 7 times 8? A:|56"
  "The first three prime numbers are|2.*3.*5"
  "Q: A train leaves at 3pm and travels for 4 hours. What time does it arrive? A:|7|seven"
  "The sun rises in the|east"
)

PASS=0; FAIL=0
for t in "${tests[@]}"; do
    prompt="${t%%|*}"
    expect="${t#*|}"
    resp=$(curl -s --max-time 60 "$BASE_URL/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"$prompt\",\"max_tokens\":48,\"temperature\":0}" \
        | /opt/venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["text"].strip()[:120].replace("\n"," ¶ "))' \
        2>/dev/null || echo "[REQUEST FAILED]")
    if echo "$resp" | grep -qiE "$expect"; then
        echo "  ✓ PASS  prompt='$prompt'"
        echo "          expected~/$expect/  got=\"$resp\""
        PASS=$((PASS+1))
    else
        echo "  ✗ FAIL  prompt='$prompt'"
        echo "          expected~/$expect/  got=\"$resp\""
        FAIL=$((FAIL+1))
    fi
done
echo
echo "[sanity] $PASS passed, $FAIL failed (of $((PASS+FAIL)) tests)"
exit $((FAIL > 0))
