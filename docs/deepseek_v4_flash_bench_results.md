# DeepSeek‑V4‑Flash — Benchmark Results (ROCm)

Date: 2026‑05‑29
Hardware: 8× AMD GPU (gfx1201), TP=8
Engine: vLLM, `cudagraph_mode = PIECEWISE`, kv‑cache fp8
Tool: `vllm bench serve` (random dataset)

Run config: `num_prompts=10`, `input_len=1024`, `output_len=1024`,
`request_rate=inf`, `ignore_eos=1`, `num_warmups=10`.
Each concurrency point is the average of 2 runs.

## Median latencies vs concurrency

| Concurrency | Output tok/s | Total tok/s | Req/s | Median TTFT (ms) | Median ITL (ms) | Median TPOT (ms) | Median E2E (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 5.32 | 10.64 | 0.0052 | 1795.1 | 187.0 | 186.3 | 192389.3 |
| 2 | 9.22 | 18.44 | 0.0090 | 2309.2 | 214.8 | 214.4 | 221987.9 |
| 4 | 13.32 | 26.64 | 0.0130 | 4809.3 | 252.9 | 261.8 | 272873.3 |

Notes:
- *Output tok/s*, *Total tok/s*, *Req/s* are aggregate server throughput over the
  run (no median variant — they are totals).
- TTFT / ITL / TPOT / E2E are per‑request **median** values (`median_*_ms`).
- Per‑request decode rate = 1000 / TPOT → ~5.4 tok/s (c=1), ~4.7 (c=2),
  ~3.8 (c=4).
- The `input_len=2048 / output_len=2048` runs completed 0 requests
  (failed/aborted) and are excluded.

## Takeaways

- Throughput scales sublinearly with concurrency while latency degrades
  gradually: 1→2 gives +73% output tok/s for +15% ITL; 1→4 gives +150% for +35%
  ITL.
- c=1 is lowest latency / worst utilization; higher concurrency trades latency
  for throughput. The "best" point depends on the latency SLA.
- All points are slow in absolute terms (~4–5 tok/s/user). The dominant cost is
  the eager ROCm sparse‑attention fallback (`_mqa_logits_paged_torch` Python
  per‑block loop + the un‑fused MLA chunk loop), which runs eagerly outside the
  CUDA graph under PIECEWISE. Concurrency tuning cannot fix this; a fused Triton
  kernel is the real win (see deepseek_v4_flash_rocm_cudagraph_fix.md follow‑ups).
