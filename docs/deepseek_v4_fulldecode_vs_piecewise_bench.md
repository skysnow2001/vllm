# DeepSeek-V4-Flash: FULL_DECODE_ONLY vs PIECEWISE cudagraph (gfx12 / 8×Navi48)

Decode-step CUDA-graph capture comparison on 8× gfx1201 (TP=8), DeepSeek-V4-Flash
(FP8 KV, MXFP4 MoE). All runs are the **same benchmark config** — apples-to-apples.

## Benchmark config
- requests: 10, **max concurrency: 1**
- input: 1024 tok/req, output: 1024 tok/req (10240 in / 10240 out total)
- served via `scripts/serve_deepseek_v4_flash.sh`, bench via the serve benchmark client

## Results

| Metric | PIECEWISE (baseline) | **FULL_DECODE_ONLY** | Speedup |
|---|---|---|---|
| Median TPOT (decode / token) | 186.3 ms | **47.6 ms** | **~3.9×** |
| Median ITL | 186.7 ms | **47.6 ms** | **~3.9×** |
| Output token throughput | 5.32 tok/s | **20.82 tok/s** | **~3.9×** |
| Median TTFT | ~1795 ms | **480 ms** | **~3.7×** |
| E2E latency (1024-tok req) | ~192 s | **~49 s** | **~3.9×** |

### FULL_DECODE_ONLY (this branch, gfx12)
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260608T154900Z.json`
- output 20.82 tok/s · median TPOT 47.63 ms · median ITL 47.55 ms · median TTFT 480.44 ms
- `CUDAGRAPH_MODE=FULL_DECODE_ONLY`, `VLLM_DSV4_TRITON=1`, `MOE_BACKEND=triton_unfused`

### PIECEWISE baseline (`/app/vllm_deepseek`, branch 6e503868c, 05-29)
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260529T175530Z.json` → 5.33 tok/s · TPOT 186.27 ms · TTFT 1803 ms
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260529T192149Z.json` → 5.31 tok/s · TPOT 186.36 ms · TTFT 1787 ms
- (two runs agree closely → stable baseline)

## Why ~4×
At concurrency=1 the decode step is **CPU launch-bound**: a 43-layer model with
sparse-MLA + Lightning indexer + MoE issues a very large number of small kernels
per token.
- **PIECEWISE**: graph is split at every attention boundary → many graph-segment
  replays with attention/indexer run eagerly between them → large per-token host
  overhead.
- **FULL_DECODE_ONLY**: the entire decode forward is captured as **one** cudagraph
  → a single replay per token with almost no launch overhead.

Median decode latency dropping 186 ms → 48 ms (~139 ms/token of host overhead
removed) is consistent with eliminating per-launch overhead for a launch-bound
decode.

## Caveat — not a pure full-vs-piecewise isolation
The PIECEWISE baseline is from a **different workspace/branch** (`6e503868c`),
which may differ from this build in more than cudagraph mode (e.g. some
torch-fallback kernels vs. the Triton kernels wired on this branch). So the ~3.9×
reflects **full-graph capture + any kernel/version differences combined**, not
full-vs-piecewise in isolation.

To isolate the pure cudagraph-mode effect, run the **current** build with
`CUDAGRAPH_MODE=PIECEWISE` (same conc=1, 1024/1024) and compare to the 47.6 ms
TPOT above.
