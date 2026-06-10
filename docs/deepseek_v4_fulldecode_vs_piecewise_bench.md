# DeepSeek-V4-Flash: FULL_DECODE_ONLY vs PIECEWISE cudagraph (gfx12 / 8×Navi48)

Decode-step CUDA-graph capture comparison on 8× gfx1201 (TP=8), DeepSeek-V4-Flash
(FP8 KV, MXFP4 MoE). All runs are the **same benchmark config** — apples-to-apples.

## Benchmark config
- requests: 10, **max concurrency: 1**
- input: 1024 tok/req, output: 1024 tok/req (10240 in / 10240 out total)
- served via `scripts/serve_deepseek_v4_flash.sh`, bench via the serve benchmark client

## Results

| Metric | PIECEWISE (baseline) | **FULL_DECODE_ONLY** | **FULL_AND_PIECEWISE** | Speedup (FDO vs PW) |
|---|---|---|---|---|
| Median TPOT (decode / token) | 186.3 ms | **43.96 ms** | 44.00 ms | **~4.2×** |
| Median ITL | 186.7 ms | **43.91 ms** | 43.87 ms | **~4.3×** |
| Output token throughput | 5.32 tok/s | **22.55 tok/s** | 22.50 tok/s | **~4.2×** |
| Median TTFT | ~1795 ms | 445 ms | **428 ms** | ~4.0× |
| E2E latency (1024-tok req) | ~192 s | **~45.4 s** | ~45.4 s | **~4.2×** |

**FULL_DECODE_ONLY vs FULL_AND_PIECEWISE:** decode is identical (same single full
decode graph) → TPOT / ITL / output throughput unchanged. `FULL_AND_PIECEWISE`
only adds piecewise capture of the **prefill**, which buys ~4–6% lower TTFT
(median 445 → 428 ms, mean 428 → 403 ms) at the cost of a second capture set
(more warmup + VRAM). At conc=1 / 1024-out the TTFT win washes out in E2E
(decode-bound), so `FULL_DECODE_ONLY` is the leaner choice; prefer
`FULL_AND_PIECEWISE` for TTFT-sensitive / long-prompt / many-concurrent-prefill
workloads.

### FULL_DECODE_ONLY (this branch, gfx12) — latest
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260610T154655Z.json`
- output 22.55 tok/s · median TPOT 43.96 ms · median ITL 43.91 ms · median TTFT 445.37 ms · median E2E 45.41 s
- `CUDAGRAPH_MODE=FULL_DECODE_ONLY`, `VLLM_DSV4_TRITON=1`, `MOE_BACKEND=triton_unfused`,
  `VLLM_ROCM_USE_AITER=1` (AITER on; native sampler on gfx12)
- Uses the **fused C++/HIP qnorm+rope+kv-insert `_C` kernel** + the **aiter sparse
  indexer** (vs the earlier torch-reference qnorm + custom Triton indexer).

### FULL_AND_PIECEWISE (same build, prefill also captured)
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260610T180533Z.json`
- output 22.50 tok/s · median TPOT 44.00 ms · median ITL 43.87 ms · median TTFT 427.62 ms · median E2E 45.44 s
- `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` (decode = full graph, prefill = piecewise),
  same other env as above
- Decode identical to FULL_DECODE_ONLY; only TTFT improves (~4–6%).

### Prior FULL_DECODE_ONLY (torch-ref qnorm + custom indexer, 06-08)
- `results/deepseek-v4-flash_np10_conc1_in1024_out1024_20260608T154900Z.json`
- output 20.82 tok/s · median TPOT 47.63 ms · median TTFT 480.44 ms
- → moving qnorm to the fused `_C` kernel and the indexer to aiter gave a further
  **~8%** TPOT improvement (47.63 → 43.96 ms) on top of full-graph capture.

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

Median decode latency dropping 186 ms → 44 ms (~142 ms/token of host overhead
removed) is consistent with eliminating per-launch overhead for a launch-bound
decode.

## Caveat — not a pure full-vs-piecewise isolation
The PIECEWISE baseline is from a **different workspace/branch** (`6e503868c`),
which may differ from this build in more than cudagraph mode (e.g. some
torch-fallback kernels vs. the Triton/C++ kernels wired on this branch). So the
~4.2× reflects **full-graph capture + any kernel/version differences combined**,
not full-vs-piecewise in isolation.

To isolate the pure cudagraph-mode effect, run the **current** build with
`CUDAGRAPH_MODE=PIECEWISE` (same conc=1, 1024/1024) and compare to the 43.96 ms
TPOT above.
