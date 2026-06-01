# DeepSeek‑V4 Flash on ROCm — CUDA Graph Capture Fixes

Date: 2026‑05‑28
Branch: `main`
Goal: get `scripts/serve_deepseek_v4_flash.sh` to boot the DeepSeek‑V4‑Flash
server on ROCm (8× gfx1201) past CUDA‑graph capture.

## TL;DR

The ROCm sparse‑MLA / sparse‑attn‑indexer fallbacks contain **host‑side control
flow driven by GPU‑resident values**. Those patterns are illegal inside CUDA
(HIP) graph capture. Three distinct capture failures were hit in sequence; each
was fixed and the server now starts and serves.

Final result: server reaches `Application startup complete`, `/v1/models` and
`/v1/completions` respond.

---

## Failure 1 — `hipErrorStreamCaptureUnsupported` in the sparse MLA kernel

**Error**

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
(hipErrorStreamCaptureUnsupported)
```

**Where**

`vllm/v1/attention/ops/rocm_flash_mla_sparse.py`
- `_sparse_attn_chunked` (line ~374): `if not valid.any(): continue`
- `flash_mla_with_kvcache_rocm` inner `step()` (line ~575): `if not valid.any(): return`

**Why**

`valid.any()` returns a 0‑D GPU tensor; using it in a Python `if` forces an
implicit `.item()` → device‑to‑host sync, which HIP forbids during graph
capture.

**Fix**

Removed the two data‑dependent early‑outs (commented out in the working tree).
Correctness is preserved because `scores.masked_fill(~valid, -inf)` already
neutralises invalid (`-1`) slots, and an all‑`-inf` chunk contributes nothing to
the online‑softmax accumulators. The only cost is running a fully‑padded chunk's
math instead of skipping it.

---

## Failure 2 / 3 — `Cannot copy between CPU and CUDA tensors during CUDA graph capture`

**Error**

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```

**Where**

`vllm/v1/attention/ops/rocm_sparse_attn_indexer.py`, `_mqa_logits_paged_torch`:
- line 280: `ctx_lens = context_lens_b.tolist()`
- line 304: `phys_block = int(block_tables[i, block_rk].item())`

These are GPU→CPU copies used to drive the Python `for` loops and slice the KV
cache. They are fine in eager mode but illegal during graph capture (and the
destination Python list/int is not pinned memory).

### Why pinned memory is NOT the fix here

Pinning only lets a D2H copy be *recorded* in a graph; it does not let Python
*read* the value during capture (`.tolist()`/`.item()` still block on the GPU).
Control‑flow uses therefore cannot be rescued by pinning.

### Step A — tag the op as `cudagraph_unsafe` (necessary, not sufficient alone)

`_mqa_logits_paged_torch` is the slow‑but‑correct eager fallback (a future fused
Triton kernel is the real path). It should never be captured. Tagged it so
vLLM's piecewise compile splits the graph around it.

File: `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py` (op registration)

```python
direct_register_custom_op(
    op_name="rocm_sparse_attn_indexer_no_insert",
    op_func=rocm_sparse_attn_indexer_no_insert,
    mutates_args=["topk_indices_buffer"],
    fake_impl=rocm_sparse_attn_indexer_no_insert_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch._C.Tag.cudagraph_unsafe,),   # <-- added
)
```

Note: `vllm::rocm_sparse_attn_indexer_no_insert` is also already present in
`CompilationConfig._attention_ops` (`vllm/config/compilation.py:769`), so it is
a piecewise splitting boundary. The PIECEWISE capture therefore already ran the
op eagerly and succeeded.

### Step B — force `cudagraph_mode = PIECEWISE` (the actual fix)

The model defaulted to `cudagraph_mode = FULL_AND_PIECEWISE`. The PIECEWISE pass
succeeded, but vLLM then attempted a second **`Capturing CUDA graphs (decode,
FULL)`** pass. A FULL cudagraph captures the *entire* decode forward as one
graph — `splitting_ops` / `cudagraph_unsafe` tags do **not** exclude anything
from a FULL capture — so it re‑entered the indexer op and died on the same
GPU→CPU copy.

This op fundamentally cannot live in a FULL graph (its Python loop bounds and
per‑step block indices would be baked in at capture time, which is also
incorrect across decode steps). The fix is to run PIECEWISE‑only, which is
already proven to capture everything else and run the indexer eagerly.

Implemented in the serve script (`scripts/serve_deepseek_v4_flash.sh`):

```bash
# default; override with CUDAGRAPH_MODE=...
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
...
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
elif [[ -n "$CUDAGRAPH_MODE" ]]; then
    EXTRA_ARGS+=(--compilation-config "{\"cudagraph_mode\": \"$CUDAGRAPH_MODE\"}")
fi
```

---

## Files changed

| File | Change |
|------|--------|
| `vllm/v1/attention/ops/rocm_flash_mla_sparse.py` | Removed two `if not valid.any():` host‑sync early‑outs (lines ~374, ~575). |
| `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py` | Added `tags=(torch._C.Tag.cudagraph_unsafe,)` to the `direct_register_custom_op` call. |
| `scripts/serve_deepseek_v4_flash.sh` | Added `CUDAGRAPH_MODE` (default `PIECEWISE`) wired into `--compilation-config`. |

## Verification

```text
compilation_config ... 'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%   # only PIECEWISE, no decode-FULL
Application startup complete.
```

Hard‑error count (`Worker failed`, `EngineCore failed`, stream‑capture,
CPU↔CUDA copy): **0**.

Smoke test:

```bash
curl -s localhost:8000/v1/models            # lists deepseek-v4-flash
curl -s localhost:8000/v1/completions -d '{"model":"deepseek-v4-flash","prompt":"...","max_tokens":16}'
# -> returns a valid completion (HTTP 200)
```

## Follow‑ups (not required to boot)

1. Replace the eager `_mqa_logits_paged_torch` Python loop with a fused Triton
   kernel that walks the block table on‑device. This removes the host sync
   entirely and would allow FULL cudagraph (better decode perf).
2. Similarly, fuse the `rocm_flash_mla_sparse` einsum/online‑softmax loop into a
   single capture‑safe kernel; then the removed `valid.any()` early‑outs can be
   handled on‑device with no wasted work.
3. Until then, keep `CUDAGRAPH_MODE=PIECEWISE` for this model on ROCm.
