#!/usr/bin/env bash
# Benchmark / smoke-test the DeepSeek-V4-Flash vLLM server started by
# scripts/serve_deepseek_v4_flash.sh.
#
# Three modes (selected via MODE env var):
#
#   MODE=quick   (default if no MODE set)
#       Fire 2-3 curl smoke tests against /v1/models, /v1/completions and
#       /v1/chat/completions. Fastest way to confirm the server is alive
#       and producing tokens. ~5 s.
#
#   MODE=bench
#       Run `vllm bench serve` with the random-prompt dataset (no external
#       download required). Reports throughput, TTFT, ITL, P50/P95/P99
#       latency, etc. Tunable via NUM_PROMPTS / MAX_CONCURRENT / INPUT_LEN
#       / OUTPUT_LEN env vars.
#
#   MODE=eval
#       Run `lm_eval --model local-completions` against the server. Falls
#       back to a clear install hint if lm_eval is not installed (it isn't
#       by default — installing it pulls hundreds of MB of HF datasets
#       deps so we don't auto-install). Tunable via TASKS env var.
#
# Examples:
#   scripts/bench_deepseek_v4_flash.sh                            # quick smoke
#   MODE=bench scripts/bench_deepseek_v4_flash.sh                 # default 200-prompt random bench
#   MODE=bench NUM_PROMPTS=1000 MAX_CONCURRENT=128 scripts/bench_deepseek_v4_flash.sh
#   MODE=eval TASKS=gsm8k scripts/bench_deepseek_v4_flash.sh       # accuracy
#   MODE=eval TASKS='gsm8k,arc_challenge' scripts/bench_deepseek_v4_flash.sh

set -euo pipefail

# Force unbuffered output so the tee'd log updates in real time.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Preserve color through the tee pipe (same trick as the serve script).
export FORCE_COLOR="${FORCE_COLOR:-1}"
export CLICOLOR_FORCE="${CLICOLOR_FORCE:-1}"
export PY_COLORS="${PY_COLORS:-1}"
export COLORTERM="${COLORTERM:-truecolor}"
export TERM="${TERM:-xterm-256color}"

# --- knobs --------------------------------------------------------------
MODE="${MODE:-quick}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT}"

# Name registered with the server (matches --served-model-name in serve_*.sh).
# vllm bench serve and lm_eval both want this — not the on-disk model path.
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
# Tokenizer / "model" path on disk; vllm bench serve uses this for tokenizer
# init when computing per-token metrics.
MODEL_PATH="${MODEL_PATH:-/app/models/deepseek-ai/DeepSeek-V4-Flash}"

# bench-mode knobs
NUM_PROMPTS="${NUM_PROMPTS:-200}"
# Empty = unlimited concurrency (no client-side --max-concurrency gate; matches
# vllm bench serve's own default of None). Set MAX_CONCURRENT=N to cap in-flight
# requests for a controlled load point.
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
DATASET_NAME="${DATASET_NAME:-random}"
REQUEST_RATE="${REQUEST_RATE:-inf}"      # inf = back-to-back; or a number = req/s
NUM_WARMUPS="${NUM_WARMUPS:-10}"         # warmup reqs excluded from percentiles
IGNORE_EOS="${IGNORE_EOS:-1}"            # 1 = force OUTPUT_LEN tokens regardless of EOS
SAVE_DETAILED="${SAVE_DETAILED:-0}"      # 1 = include per-request timing in the JSON

# eval-mode knobs
TASKS="${TASKS:-gsm8k}"
EVAL_NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-64}"
EVAL_BATCH="${EVAL_BATCH:-auto}"
# LIMIT=N → cap each task to N samples; useful for quick correctness checks
# (e.g. LIMIT=20 → ~20 GSM8K problems instead of 1319). Default empty = full.
LIMIT="${LIMIT:-}"
# lm_eval needs a local tokenizer for length accounting (the model itself
# runs on the vllm server). We default to the HF repo id so AutoTokenizer
# fetches just the tokenizer files (~MB, not the model weights) the first
# time, then caches them under ~/.cache/huggingface. Override with a local
# path if you don't have network, e.g. TOKENIZER=$MODEL_PATH.
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-V4-Flash}"

# Pick vllm binary: prefer in-repo venv, fall back to PATH.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
if [[ -x "$REPO/.venv/bin/vllm" ]]; then
    VLLM_BIN="${VLLM_BIN:-$REPO/.venv/bin/vllm}"
else
    VLLM_BIN="${VLLM_BIN:-vllm}"
fi
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

# Log file (timestamped + 'latest' symlink, same pattern as serve_*.sh).
LOG_DIR="${LOG_DIR:-$REPO/logs}"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/bench_${MODE}_$TS.log}"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/bench_${MODE}_latest.log"

# Benchmark JSON results go in a dedicated results/ dir with a descriptive name
# encoding model + run params, and are pretty-printed after the run.
RESULTS_DIR="${RESULTS_DIR:-$REPO/results}"
mkdir -p "$RESULTS_DIR"
# Sanitize served name for use in a filename (slashes -> dashes).
RESULT_MODEL_TAG="$(echo "$SERVED_NAME" | tr '/ ' '--')"
# Encode params: model, num_prompts, max_concurrent, input_len, output_len.
RESULT_CONC="${MAX_CONCURRENT:-unlimited}"
RESULT_NAME="${RESULT_MODEL_TAG}_np${NUM_PROMPTS}_conc${RESULT_CONC}_in${INPUT_LEN}_out${OUTPUT_LEN}_${TS}.json"

# --- preflight: is the server up? ---------------------------------------
preflight() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
           "$BASE_URL/v1/models" 2>/dev/null || true)
    if [[ "$code" != "200" ]]; then
        echo "[bench] ERROR: server at $BASE_URL/v1/models returned HTTP $code" >&2
        echo "[bench]        is scripts/serve_deepseek_v4_flash.sh running?"    >&2
        exit 1
    fi
}

# --- mode: quick ---------------------------------------------------------
run_quick() {
    echo "[bench/quick] /v1/models"
    curl -s --max-time 5 "$BASE_URL/v1/models" \
        | python3 -m json.tool 2>/dev/null || echo "  (model list unavailable)"
    echo
    echo "[bench/quick] /v1/completions  (greedy, 16 tokens)"
    curl -s --max-time 30 "$BASE_URL/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
        | python3 -m json.tool 2>/dev/null || echo "  (completion failed)"
    echo
    echo "[bench/quick] /v1/chat/completions  (5-word hello, T=0.7)"
    curl -s --max-time 30 "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in exactly 5 words.\"}],\"max_tokens\":32,\"temperature\":0.7}" \
        | python3 -m json.tool 2>/dev/null || echo "  (chat failed)"
}

# --- mode: bench (vllm bench serve) --------------------------------------
run_bench() {
    # Compose dataset-specific args.
    local extra=()
    case "$DATASET_NAME" in
        random)
            extra+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
            ;;
        sharegpt)
            # Requires DATASET_PATH=/path/to/sharegpt.json
            if [[ -z "${DATASET_PATH:-}" ]]; then
                echo "[bench/bench] ERROR: DATASET_NAME=sharegpt needs DATASET_PATH=…/sharegpt.json" >&2
                exit 2
            fi
            extra+=(--dataset-path "$DATASET_PATH")
            ;;
        *)
            # Other dataset names (sonnet, hf, …) — caller must provide
            # whatever extra knobs are required via the environment.
            if [[ -n "${DATASET_PATH:-}" ]]; then
                extra+=(--dataset-path "$DATASET_PATH")
            fi
            ;;
    esac

    # Optional extra flags (kept as an array so empties don't expand to "").
    local opt_args=()
    [[ "$IGNORE_EOS"    == "1" ]] && opt_args+=(--ignore-eos)
    [[ "$SAVE_DETAILED" == "1" ]] && opt_args+=(--save-detailed)
    [[ "$NUM_WARMUPS"   -gt 0  ]] && opt_args+=(--num-warmups "$NUM_WARMUPS")
    # Only pass --max-concurrency when set; empty => unlimited (server-limited).
    [[ -n "$MAX_CONCURRENT" ]] && opt_args+=(--max-concurrency "$MAX_CONCURRENT")

    echo "[bench/bench] vllm bench serve"
    echo "  base_url       : $BASE_URL"
    echo "  served_name    : $SERVED_NAME"
    echo "  model (tok)    : $MODEL_PATH"
    echo "  dataset        : $DATASET_NAME"
    echo "  num_prompts    : $NUM_PROMPTS"
    echo "  max_concurrent : ${MAX_CONCURRENT:-unlimited}"
    echo "  input_len      : $INPUT_LEN  (random-mode only)"
    echo "  output_len     : $OUTPUT_LEN  (random-mode only)"
    echo "  request_rate   : $REQUEST_RATE"
    echo "  num_warmups    : $NUM_WARMUPS  (excluded from percentiles)"
    echo "  ignore_eos     : $IGNORE_EOS   (1=force OUTPUT_LEN tokens)"
    echo "  save_detailed  : $SAVE_DETAILED"
    echo

    "$VLLM_BIN" bench serve \
        --backend openai \
        --base-url "$BASE_URL" \
        --endpoint /v1/completions \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --dataset-name "$DATASET_NAME" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --metric-percentiles '50,90,95,99' \
        --save-result \
        --result-dir "$RESULTS_DIR" \
        --result-filename "$RESULT_NAME" \
        --trust-remote-code \
        "${opt_args[@]}" \
        "${extra[@]}"

    # Pretty-print the result JSON in place (vllm writes it single-line).
    local result_path="$RESULTS_DIR/$RESULT_NAME"
    if [[ -f "$result_path" ]]; then
        python3 -m json.tool "$result_path" "$result_path.tmp" \
            && mv "$result_path.tmp" "$result_path"
        echo "[bench/bench] result saved: $result_path"
    else
        echo "[bench/bench] WARNING: expected result file not found: $result_path" >&2
    fi
}

# --- mode: eval (lm_eval --model local-completions) ----------------------
run_eval() {
    if ! command -v "$LM_EVAL_BIN" >/dev/null 2>&1; then
        echo "[bench/eval] ERROR: $LM_EVAL_BIN not installed." >&2
        echo "[bench/eval] To install (pulls ~hundreds of MB of HF datasets deps):" >&2
        echo "             /opt/venv/bin/pip install 'lm_eval[api]'" >&2
        echo "             # or, with API extra for local-completions backend:"     >&2
        echo "             # /opt/venv/bin/pip install lm-eval lm-eval-harness"     >&2
        exit 3
    fi

    # `model=` in model_args has two roles in lm_eval's local-completions:
    #   1. Goes into the API request body (`{"model": "...", ...}`)
    #        → must match what vllm registered via --served-model-name
    #   2. Passed to `transformers.AutoTokenizer.from_pretrained(...)` for
    #      local prompt-tokenization / length accounting (even when
    #      tokenized_requests=False — lm_eval still needs to count tokens).
    #        → must be a valid HF repo id OR a local path
    # The served-name we use ("deepseek-v4-flash") satisfies (1) but is NOT
    # a HF repo, so (2) hits a 404. Fix: pass `tokenizer=` explicitly. By
    # default we use the HF repo id so AutoTokenizer downloads just the
    # tokenizer files (~MB, cached at ~/.cache/huggingface). Override with
    # a local model path if offline (TOKENIZER=/app/models/...).
    echo "[bench/eval] lm_eval --model local-completions"
    echo "  base_url        : $BASE_URL/v1/completions"
    echo "  served_name     : $SERVED_NAME  (used for API request body)"
    echo "  tokenizer       : $TOKENIZER  (used for local length accounting)"
    echo "  tasks           : $TASKS"
    echo "  num_concurrent  : $EVAL_NUM_CONCURRENT"
    echo "  batch_size      : $EVAL_BATCH"
    echo

    EVAL_EXTRA=()
    if [[ -n "$LIMIT" ]]; then
        EVAL_EXTRA+=(--limit "$LIMIT")
        echo "  limit            : $LIMIT samples per task (quick check)"
    fi

    "$LM_EVAL_BIN" \
        --model local-completions \
        --tasks "$TASKS" \
        --model_args "model=$SERVED_NAME,tokenizer=$TOKENIZER,base_url=$BASE_URL/v1/completions,num_concurrent=$EVAL_NUM_CONCURRENT,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
        --batch_size "$EVAL_BATCH" \
        "${EVAL_EXTRA[@]}"
}

# --- main ----------------------------------------------------------------
echo "[bench] mode=$MODE  log=$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

preflight

case "$MODE" in
    quick) run_quick ;;
    bench) run_bench ;;
    eval)  run_eval ;;
    *)
        echo "[bench] ERROR: invalid MODE='$MODE'" >&2
        echo "  must be one of: quick, bench, eval" >&2
        exit 2
        ;;
esac
