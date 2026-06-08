#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash on Navi48 with TP=8.
#
# Usage:
#   scripts/serve_deepseek_v4_flash.sh
#
# Notes:
# - The model lives under /app/models/deepseek-ai/DeepSeek-V4-Flash.
# - Adjust HOST/PORT/TP/MAX_MODEL_LEN via env vars.

set -euo pipefail

#clean graph compile
rm -rf /root/.cache/vllm/torch_compile_cache/torch_aot_compile 
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/z3/lib:${LD_LIBRARY_PATH}"

# Force unbuffered Python stdout/stderr so the tee'd log file updates in
# real time instead of in 4-8 KiB chunks.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Tee'ing through a pipe makes vllm/Python see stdout as non-TTY and strip
# ANSI colors. Force-enable color output across the common conventions so
# the terminal stays colored (the log file will then contain raw escape
# sequences — strip with `sed 's/\x1b\[[0-9;]*m//g'` if you need plain text).
export FORCE_COLOR="${FORCE_COLOR:-1}"
export CLICOLOR_FORCE="${CLICOLOR_FORCE:-1}"
export PY_COLORS="${PY_COLORS:-1}"
export COLORTERM="${COLORTERM:-truecolor}"
# Some libs only check TERM != "dumb"; carry the parent's TERM through.
export TERM="${TERM:-xterm-256color}"

MODEL_PATH="${MODEL_PATH:-/app/models/deepseek-ai/DeepSeek-V4-Flash}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
# Pipeline-parallel size. Set PP>1 when the model is too large to fit at the
# desired TP (i.e. weights / TP > per-GPU memory) AND the layer count is
# divisible by PP. Per-GPU weight footprint becomes ~total_weights/(PP*TP).
# Total GPUs used = PP * TP. Common: PP=2 TP=4 to use 8 GPUs while keeping
# per-GPU sharding compatible with FP8 block_n=128.
PP="${PP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
DTYPE="${DTYPE:-auto}"
# DeepSeek V4 enforces fp8 kv-cache (asserted in DeepseekV4MLAAttention.__init__);
# vllm will auto-promote to fp8_ds_mla for the FlashMLA Sparse backend.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# ENFORCE_EAGER=1 disables torch.compile / Inductor / CUDA-graph capture.
# Use it when bringing up new code or when the AOT artifact loader deadlocks
# (see py-spy: stuck in vllm/compilation/caching.py:load_all on ROCm). Costs
# perf but boots in ~2 min with no Triton/Inductor surface area.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

# CUDA graph capture mode. The ROCm sparse-attn indexer fallback
# (rocm_sparse_attn_indexer_no_insert -> _mqa_logits_paged_torch) drives its
# block-table walk with Python loops over GPU-resident values, so it cannot be
# captured inside a *FULL* cudagraph (the default FULL_AND_PIECEWISE tries a
# "decode, FULL" capture that hits a GPU->CPU copy and dies). PIECEWISE splits
# the graph at that op (it is listed in CompilationConfig._attention_ops) and
# runs it eagerly, which is correct and already captures everything else.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"

# Which MXFP4 MoE backend to pass to `vllm serve --moe-backend`.
# Dispatch table at vllm/model_executor/layers/fused_moe/oracle/mxfp4.py.
#
#   auto            → no flag passed; vllm's oracle walks its own priority
#                     list and picks the first supported backend (default).
#   triton          → Mxfp4MoeBackend.TRITON         → OAITritonMxfp4ExpertsMonolithic
#                     OpenAI `triton_kernels` fused matmul.
#   triton_unfused  → Mxfp4MoeBackend.TRITON_UNFUSED → UnfusedOAITritonExperts
#                     OpenAI `triton_kernels` modular path; the DSv4-Flash
#                     MoE path used on gfx12. Routing goes through the OAI
#                     `triton_kernels.routing` kernel, which needs the top-k=6
#                     power-of-2 patch in
#                     vllm/third_party/triton_kernels/routing_details/_routing_compute.py
#                     (otherwise `tl.arange(0, N_EXPTS_ACT*BLOCK_M)` asserts).
#
# When MOE_BACKEND=auto we DON'T forward --moe-backend (vllm's CLI default is
# also "auto"); that way the oracle takes the "auto" code path with
# platform-specific move-to-front rules instead of the explicit-override
# branch. Any other value is forwarded to --moe-backend unchanged.
MOE_BACKEND="${MOE_BACKEND:-auto}"
case "$MOE_BACKEND" in
    auto|triton|triton_unfused) ;;
    *)
        echo "[serve_deepseek_v4_flash] ERROR: invalid MOE_BACKEND='$MOE_BACKEND'" >&2
        echo "  must be one of: auto, triton, triton_unfused" >&2
        exit 2
        ;;
esac

# ROCm / AITER toggles (defaults tuned for Navi48 / gfx12).
#
# VLLM_ROCM_USE_AITER_MOE is FORCED (not defaulted) based on $MOE_BACKEND.
# We can't use ${VAR:-default} here because that honors an existing VAR=0
# from the user's shell. For triton / triton_unfused we hard-set both AITER
# and AITER_MOE to 0 so AITER doesn't get pulled into unrelated code paths.
_force_env() {
    # _force_env NAME VALUE → export NAME=VALUE, warn if it overrode a
    # conflicting inherited value.
    local name="$1" want="$2" had="${!1-}"
    if [[ -n "${!1+set}" && "$had" != "$want" ]]; then
        echo "[serve_deepseek_v4_flash] WARN: overriding $name=$had -> $want (required for MOE_BACKEND=$MOE_BACKEND)" >&2
    fi
    export "$name=$want"
}
# Force env only when MOE_BACKEND demands a specific AITER answer:
#   triton | triton_unfused     → AITER explicitly NOT wanted; force both to 0
#                                 so the oracle removes it from the candidate
#                                 list (avoids noisy "AITER rejected" logs).
#   auto                        → DON'T touch the env vars. vllm's defaults
#                                 (AITER=False, AITER_MOE=True) let the
#                                 oracle's normal is_supported_config() gate
#                                 decide based on the actual GPU arch.
case "$MOE_BACKEND" in
    triton|triton_unfused)
        # AITER not wanted here: MoE uses OAI triton_kernels, and the DSv4
        # Triton paths (blockscale GEMM, MHC, indexer, sparse-MLA, o-proj) are
        # direct triton imports enabled by VLLM_DSV4_TRITON — not gated by
        # rocm_aiter_ops.is_enabled(). Keeping AITER on pulls in the aiter
        # sampler (top_k_top_p_sampling_from_probs) whose JIT .so fails to build
        # on gfx12, crashing the profile run. Force both off → native sampler.
        _force_env VLLM_ROCM_USE_AITER     0
        _force_env VLLM_ROCM_USE_AITER_MOE 0
        ;;
    auto)
        : ;;  # intentionally no force — let vllm decide
esac
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-0}"
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-1}"
# NOTE: the fused Triton sparse MLA backend (rocm_aiter_mla_sparse, PR #41812)
# is the only ROCm sparse-MLA path (no env var needed).

# Pick vllm binary: prefer in-repo venv, fall back to PATH.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
if [[ -x "$REPO/.venv/bin/vllm" ]]; then
    VLLM_BIN="${VLLM_BIN:-$REPO/.venv/bin/vllm}"
else
    VLLM_BIN="${VLLM_BIN:-vllm}"
fi

# Tee stdout+stderr to logs/serve_deepseek_v4_flash-<UTC-timestamp>.log so the
# console stays interactive while a complete copy lands on disk.
LOG_DIR="${LOG_DIR:-$REPO/logs}"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/serve_deepseek_v4_flash-$TS.log}"
# Also expose as a "latest" symlink for convenience.
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/serve_deepseek_v4_flash-latest.log"

# Print the banner to a function so we can call it AFTER the tee redirect
# (otherwise the banner only goes to the terminal, not to $LOG_FILE — which
# made early bring-up debugging painful: we couldn't see which env vars the
# script resolved when a crash happened).
print_banner() {
    echo "[serve_deepseek_v4_flash] vllm bin      : $VLLM_BIN"
    echo "[serve_deepseek_v4_flash] model        : $MODEL_PATH"
    echo "[serve_deepseek_v4_flash] tp size      : $TP"
    echo "[serve_deepseek_v4_flash] pp size      : $PP  (total GPUs used: $((TP*PP)))"
    echo "[serve_deepseek_v4_flash] max_model_len: $MAX_MODEL_LEN"
    echo "[serve_deepseek_v4_flash] kv_cache     : $KV_CACHE_DTYPE"
    echo "[serve_deepseek_v4_flash] enforce_eager: $ENFORCE_EAGER"
    if [[ "$MOE_BACKEND" == "auto" ]]; then
        echo "[serve_deepseek_v4_flash] moe_backend  : auto (no --moe-backend passed; vllm oracle picks)"
    else
        echo "[serve_deepseek_v4_flash] moe_backend  : $MOE_BACKEND (forwarded as --moe-backend)"
    fi
    echo "[serve_deepseek_v4_flash] AITER env    : USE_AITER=${VLLM_ROCM_USE_AITER:-(unset)}  AITER_MOE=${VLLM_ROCM_USE_AITER_MOE:-(unset)}  RMSNORM=${VLLM_ROCM_USE_AITER_RMSNORM:-(unset)}  UNIFIED_ATTN=${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-(unset)}"
    echo "[serve_deepseek_v4_flash] host:port    : $HOST:$PORT"
    echo "[serve_deepseek_v4_flash] log file     : $LOG_FILE"
}

# --- cleanup trap ----------------------------------------------------------
#
# vLLM v1 spawns TP workers (and a multiprocessing resource_tracker) via
# Python's `multiprocessing` machinery, which does NOT set
# `prctl(PR_SET_PDEATHSIG, …)` on the children. If this script (or the
# EngineCore parent) is hard-killed while a worker is stuck inside a C/Triton
# extension — e.g. `torch._inductor.runtime.autotune_cache._get` or the AOT
# Triton launcher — the worker won't see SIGTERM until it returns to Python,
# which may be never. The orphaned workers re-parent to PID 1 and keep
# holding GPU memory (KFD reports them as "UNKNOWN" PID), which then
# blocks the next launch with "Free memory on device cuda:X (10/31.86 GiB)
# is less than desired GPU memory utilization".
#
# To make Ctrl-C / SIGTERM actually clean up the whole tree:
#   1. Launch vllm under `setsid` so it becomes the leader of a new session
#      and process group (PGID == vllm's PID).
#   2. Run vllm in the background and `wait` on it so this script stays
#      alive to service signal traps.
#   3. On any exit path, send SIGTERM to the WHOLE process group
#      (`kill -- -PGID`), wait up to 10 s for graceful shutdown, then
#      SIGKILL the group. Finally `pkill` any stragglers as belt-and-braces.
VLLM_PID=""
cleanup() {
    local ec=$?
    trap - EXIT INT TERM HUP
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[serve_deepseek_v4_flash] stopping pgrp $VLLM_PID (TERM)…" >&2
        kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[serve_deepseek_v4_flash] TERM timed out; sending KILL" >&2
            kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
        fi
    fi
    # Belt-and-braces: reap any vLLM children that escaped the pgrp
    # (e.g. orphans from a previous run, or a worker that double-forked).
    pkill -9 -f 'VLLM::Worker'                      2>/dev/null || true
    pkill -9 -f 'VLLM::EngineCore'                  2>/dev/null || true
    pkill -9 -f 'multiprocessing\.resource_tracker' 2>/dev/null || true
    exit "$ec"
}
trap cleanup EXIT INT TERM HUP

# stdbuf -oL -eL forces line-buffered stdio on the child even though it's
# attached to a pipe (tee); PYTHONUNBUFFERED handles the Python side.
# `tee -a` appends and flushes per write.
exec > >(tee -a "$LOG_FILE") 2>&1

# Banner AFTER the tee redirect so it lands in BOTH terminal and log file.
print_banner

# Optional extra args appended conditionally (kept as an array so empty
# values don't turn into a stray "" positional that vllm would reject).
# When MOE_BACKEND=auto we omit the flag entirely so vllm's oracle runs the
# full priority loop (auto branch). Passing --moe-backend=auto explicitly
# would work but is noisier in the logs and unnecessary.
EXTRA_ARGS=()
if [[ "$MOE_BACKEND" != "auto" ]]; then
    EXTRA_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
elif [[ -n "$CUDAGRAPH_MODE" ]]; then
    EXTRA_ARGS+=(--compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}")
fi

# `setsid` creates a new session whose PGID == the new program's PID, so
# `$!` below is the PGID we'll signal in cleanup(). `wait` is interruptible
# by bash signal traps, so Ctrl-C / SIGTERM here will fire cleanup().
setsid stdbuf -oL -eL "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --pipeline-parallel-size "$PP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}" &
VLLM_PID=$!

# `wait` can return non-zero (e.g. when interrupted by a signal); don't let
# `set -e` short-circuit before we record the exit code for cleanup().
set +e
wait "$VLLM_PID"
ec=$?
set -e
exit "$ec"
