# DeepSeek-V4 Operation → Backend Matrix (CUDA vs ROCm-CDNA vs gfx12 wiring)

This document inventories every **logical operation** in the DeepSeek-V4
(DSv4-Flash-FP8) forward pass — at the granularity of *mechanisms*
(MoE, attention, indexer, compressor, MHC, o-proj, …), not individual GEMMs —
and records, for each:

- **CUDA**: what the upstream NVIDIA path uses.
- **ROCm / CDNA (MI3xx, gfx942/950)**: what the stock ROCm path uses.
- **gfx12 wiring (this branch)**: what our patches make it run on Navi48
  (gfx1201), which has **no CK / ASM / TileLang / FlashMLA / DeepGEMM** backends.

Legend for backend kinds: **CK** = AITER composable-kernel HIP `.so`;
**ASM** = hand-tuned assembly; **CuteDSL/DeepGEMM/FlashMLA** = NVIDIA-only
libraries; **TileLang** = tilelang-codegen (no HIP backend); **Triton** = Triton
JIT; **torch** = eager PyTorch reference.

Two env vars introduced/used by this branch:
- `VLLM_ROCM_USE_V4_TRITON_FALLBACK` (default **False**): when True, swaps the
  sparse-MLA attention back to a torch online-softmax reference (bisection only).
- `VLLM_DSV4_TRITON` (default **False**): master "all-Triton" switch — forces
  Triton variants of aiter ops (linear/blockscale GEMM, o-proj) over CK/ASM that
  don't exist on gfx12, while keeping the Triton sparse indexer selected.

---

## Summary table

| # | Operation | CUDA backend | ROCm / CDNA (MI3xx) backend | gfx12 wiring (this branch) | Key files |
|---|-----------|--------------|------------------------------|-----------------------------|-----------|
| 1 | **MLA sparse attention** (prefill + decode QK·softmax·PV over top-k KV) | FlashMLA (`flash_mla_sparse_fwd` / `flash_mla_with_kvcache`), NVIDIA-only | Same FlashMLA path is unavailable; AITER Triton sparse-MLA backend | **Triton** (default): `DeepseekV4ROCMAiterMLASparseImpl` → `rocm_sparse_attn_prefill` / `rocm_sparse_attn_decode` (fused `@triton.jit`, in-kernel FP8 dequant + online softmax + attn_sink). Torch online-softmax backup behind `VLLM_ROCM_USE_V4_TRITON_FALLBACK=1` | CUDA impl: `vllm/models/deepseek_v4/nvidia/flashmla.py:131,294,424`; selector: `vllm/models/deepseek_v4/attention.py:79`; ROCm impl: `vllm/models/deepseek_v4/amd/rocm.py:600,719,845`; Triton kernels: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1638,1687`; torch backup: `vllm/v1/attention/ops/rocm_flash_mla_sparse.py:400,484` |
| 2 | **Lightning Indexer — prefill MQA-logits + top-k** (select top-k tokens) | DeepGEMM `fp8_fp4_mqa_logits` + `top_k_per_row_prefill` | DeepGEMM unavailable; AITER `fp8_mqa_logits` (gluon/CK) | **Triton** (default): `rocm_sparse_attn_indexer_no_insert` → `_mqa_logits_triton` (streaming `@triton.jit`) + `top_k_per_row_prefill` (C++). Forced via `VLLM_DSV4_TRITON` even when aiter enabled | CUDA: `vllm/model_executor/layers/sparse_attn_indexer.py:82,233` (`fp8_fp4_mqa_logits`); dispatch: `vllm/model_executor/layers/sparse_attn_indexer.py:506,525`; Triton: `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py:149,339` |
| 3 | **Lightning Indexer — paged-decode MQA-logits + top-k** | DeepGEMM `fp8_fp4_paged_mqa_logits` | AITER `pa_mqa_logits` (gluon, gfx942/950/1250-only) | **torch** (`_mqa_logits_paged_torch`, Python block-table loop) — **still the gap**; aiter paged-logits is gluon/CDNA-only, no gfx12 Triton dispatch | CUDA: `vllm/model_executor/layers/sparse_attn_indexer.py:314` (`fp8_fp4_paged_mqa_logits`); torch: `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py:240,473` |
| 4 | **KV compressor** (compress → RMSNorm → RoPE → quant → store to KV cache) | CuteDSL `compress_norm_rope_store_cutedsl` | Triton `compress_norm_rope_store_triton` | **Triton** (unchanged — already arch-generic) | dispatch: `vllm/models/deepseek_v4/compressor.py:338,347,351,355`; Triton kernel: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:31`; state store: `vllm/models/deepseek_v4/common/ops/save_partial_states.py:9` |
| 5 | **MHC pre** (RMSNorm + 3-stream mix projection + sigmoid/sigmoid/Sinkhorn + layer_input reduce) | TileLang `mhc_pre_tilelang` | CK `mhc_pre_aiter` (disabled — accuracy bug) → torch | **Triton** (this branch): AITER fused Triton `mhc_pre_aiter_triton` (`aiter.ops.triton.fusions.mhc.mhc`); torch backup | layer: `vllm/model_executor/layers/mhc.py:74,122` ; adapter: `vllm/model_executor/kernels/mhc/aiter_triton.py:42,54` |
| 6 | **MHC post** (mix layer output with residual streams) | TileLang `mhc_post_tilelang` | CK `mhc_post_aiter` (disabled) → torch | **Triton** (this branch): `mhc_post_aiter_triton` (`…fusions.mhc.mhc_post`); torch backup | layer: `vllm/model_executor/layers/mhc.py:207,231`; adapter: `vllm/model_executor/kernels/mhc/aiter_triton.py:115,121` |
| 7 | **MHC fused post+pre** (post then next-layer pre in one shot) | TileLang `mhc_fused_post_pre_tilelang` | torch (compose post+pre torch refs) | **Triton** (this branch): `mhc_fused_post_pre_aiter_triton` (`…fusions.mhc.mhc_post_pre`); torch-compose backup | layer: `vllm/model_executor/layers/mhc.py:393,433`; adapter: `vllm/model_executor/kernels/mhc/aiter_triton.py:149,167` |
| 8 | **HC head** (hypercompressed vocab gate reduce, MTP logits) | TileLang `hc_head_fused_kernel_tilelang` | Triton `hc_head_triton` | **Triton** (unchanged — already Triton, ported from ATOM) | layer: `vllm/model_executor/layers/mhc.py:292,306,322`; kernel: `vllm/model_executor/kernels/mhc/triton.py:143`; caller: `vllm/models/deepseek_v4/amd/mtp.py:124,127` |
| 9 | **O-projection** (inverse-RoPE + per-group `wo_a` grouped GEMM + `wo_b`) | `fused_inv_rope_fp8_quant` (Triton) + `fp8_einsum` (DeepGEMM) | torch: `rocm_inv_rope_einsum` (`rotary_emb.forward_native` + `torch.einsum`) | **Triton** (this branch, under `VLLM_DSV4_TRITON`): inverse-RoPE ref + aiter `batched_gemm_bf16` grouped GEMM (`_rocm_triton_grouped_oproj_gemm`); torch einsum backup | call: `vllm/models/deepseek_v4/attention.py:413,425,444`; ROCm path: `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:897,930,954` |
| 10 | **qnorm + RoPE + KV insert** (per-head Q RMSNorm + GPT-J RoPE + FP8 quant K write) | C++ `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | C++ kernel is CUDA-only → torch reference | **torch** ref (`_deepseek_v4_qnorm_rope_kv_insert_reference`), with Triton `quantize_and_insert_k_cache` for the cache write — **partially Triton, q-norm+rope still torch** | dispatch: `vllm/models/deepseek_v4/attention.py:602,650,663`; cache write Triton: `vllm/models/deepseek_v4/common/ops/cache_utils.py:142` |
| 11 | **Indexer Q projection + RoPE + FP8 quant** | Triton `fused_indexer_q_rope_quant` | Triton (same) | **Triton** (unchanged) | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:284` |
| 12 | **Q/KV RMSNorm (fused, weighted)** | Triton `fused_q_kv_rmsnorm` | Triton (same) | **Triton** (unchanged) | `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py:57` |
| 13 | **MTP input / shared-head RMSNorm** | Triton `fused_mtp_input_rmsnorm` / `mtp_shared_head_rmsnorm` | Triton (same) | **Triton** (unchanged) | `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:124,153` |
| 14 | **KV-cache dequant + gather** (prefill bf16 pool build) | Triton `dequantize_and_gather_k_cache` | Triton (same) | **Triton** (unchanged) | `vllm/models/deepseek_v4/common/ops/cache_utils.py:307,353` |
| 15 | **Top-k / SWA index plumbing** (combine swa+topk, ragged indptr, global topk) | n/a (FlashMLA handles internally) | Triton helpers | **Triton** (unchanged) | `vllm/models/deepseek_v4/amd/rocm.py:53,118,236,309,370` |
| 16 | **MoE experts** (routing + grouped W4 expert GEMMs, MXFP4) | DeepGemm/TrtLlm/FlashInfer FP4 experts | CK `AiterExperts` (W4A16/W4A4, gfx950-only) | **Triton** (this branch): AITER W4A8 `AiterW4A8ExpertsMonolithic` (`aiter.ops.triton.moe_routing.routing` + `moe_op_gemm_a8w4`); oracle skips CK variants on gfx12 | oracle: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:77,194,252`; experts: `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py:27,74,113,166` |
| 17 | **Dense FP8 block-scale linears** (fused_wqa_wkv, wq_b, wo_b, gate) | CuteDSL/DeepGEMM or cutlass `_scaled_mm` | CK `gemm_a8w8_blockscale` (or Triton when tuned) | **Triton** (this branch, under `VLLM_DSV4_TRITON`): forces `AiterFp8BlockScaledMMKernel.use_triton=True` → `triton_gemm_a8w8_blockscale` (`w8a8_triton_block_scaled_mm`) | kernel: `vllm/model_executor/kernels/linear/scaled_mm/aiter.py:281,334`; flag: `vllm/_aiter_ops.py:1419,1534`; Triton mm: `vllm/model_executor/layers/quantization/utils/fp8_utils.py:718,841` |
| 18 | **RMSNorm (generic transformer layernorm)** | C++ `torch.ops._C.rms_norm` (or AITER) | C++ `_C.rms_norm` (AITER RMSNorm off by default) | **C++/HIP** (not Triton — `VLLM_ROCM_USE_AITER_RMSNORM=0`; not portable target) | `vllm/model_executor/layers/layernorm.py:104,173` |
| 19 | **Token embedding / sampler / TP all-reduce** | native / RCCL | native / RCCL | **native / RCCL** (inherently not Triton) | — |

---

## Per-operation detail

### 1. MLA sparse attention
- **CUDA** uses FlashMLA's `flash_mla_sparse_fwd` (prefill) and the V4-extended
  `flash_mla_with_kvcache` (decode), imported in
  `vllm/models/deepseek_v4/nvidia/flashmla.py:26`. Neither exists in the ROCm
  `vllm._flashmla_C` build.
- **gfx12 wiring**: `_select_v4_sparse_impl()`
  (`vllm/models/deepseek_v4/attention.py:79`) returns the AITER Triton impl
  unless `VLLM_ROCM_USE_V4_TRITON_FALLBACK=1`. The fused Triton kernels live in
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` (`@triton.jit` at
  lines 1001/1083/1192; entry points `rocm_sparse_attn_prefill:1638`,
  `rocm_sparse_attn_decode:1687`). They dequant FP8 KV in-kernel and run online
  softmax with the per-head `attn_sink`.
- **Backup** (`VLLM_ROCM_USE_V4_TRITON_FALLBACK=1`): torch online-softmax in
  `vllm/v1/attention/ops/rocm_flash_mla_sparse.py:400,484` (only the FP8
  dequant-gather helper there is Triton, `:79`).

### 2–3. Lightning Indexer (MQA-logits + top-k)
- **CUDA**: DeepGEMM `fp8_fp4_mqa_logits` (prefill) and
  `fp8_fp4_paged_mqa_logits` (decode), imported at
  `vllm/model_executor/layers/sparse_attn_indexer.py:16-17`.
- **gfx12 prefill (#2)**: Triton streaming kernel `_mqa_logits_triton`
  (`vllm/v1/attention/ops/rocm_sparse_attn_indexer.py:149`), orchestrated by
  `rocm_sparse_attn_indexer_no_insert:339`. Selected at
  `…/sparse_attn_indexer.py:506` (`forward_hip`); `VLLM_DSV4_TRITON` keeps it
  even when aiter is otherwise enabled (`:525`).
- **gfx12 decode (#3)**: still `_mqa_logits_paged_torch`
  (`vllm/v1/attention/ops/rocm_sparse_attn_indexer.py:240`) — a Python
  block-table loop with host syncs. **Not yet Triton**; aiter's paged
  MQA-logits (`aiter/ops/triton/attention/pa_mqa_logits.py`) routes to a gluon
  kernel that asserts gfx942/950 and has no gfx12 path.

### 4. KV compressor
- **CUDA** uses a CuteDSL kernel; **ROCm/gfx12** use the Triton
  `compress_norm_rope_store_triton`. Dispatch is by `current_platform.is_cuda()`
  at `vllm/models/deepseek_v4/compressor.py:338`; no change needed for gfx12.

### 5–8. MHC (multi-head hyper-compression)
- **CUDA**: TileLang ops (`mhc_pre_tilelang`, etc.). TileLang has no HIP codegen,
  so `HAS_TILELANG` is forced False on ROCm
  (`vllm/model_executor/layers/mhc.py:19`).
- **Stock ROCm**: a CK path (`mhc_pre_aiter`/`mhc_post_aiter`) existed but is
  commented out for an accuracy bug, leaving torch.
- **gfx12 wiring (this branch)**: `HAS_AITER_TRITON_MHC`
  (`…/mhc.py:24`) gates the AITER **fused Triton** kernels
  (`aiter.ops.triton.fusions.mhc`) via the adapter
  `vllm/model_executor/kernels/mhc/aiter_triton.py` (`mhc_pre:42`, `mhc_post:115`,
  `mhc_post_pre:149`). torch references remain as backup. `hc_head` (#8) was
  already Triton (`…/kernels/mhc/triton.py:143`), ported from ATOM.

### 9. O-projection
- **CUDA**: `fused_inv_rope_fp8_quant` (Triton) + `fp8_einsum` (DeepGEMM) at
  `vllm/models/deepseek_v4/attention.py:425,444`.
- **Stock ROCm**: `rocm_inv_rope_einsum`
  (`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:897`) — inverse-RoPE via
  `rotary_emb.forward_native` (torch) + `torch.einsum("tgd,grd->tgr")`.
- **gfx12 wiring**: under `VLLM_DSV4_TRITON`, the grouped GEMM is routed to
  aiter `batched_gemm_bf16` via `_rocm_triton_grouped_oproj_gemm`
  (`…/rocm_aiter_mla_sparse.py:954`); an explicit launch config is passed
  because aiter ships no `gfx1201-BATCHED_GEMM-A16W16` tuning json. Falls back to
  einsum on any error. (Inverse-RoPE itself is still the torch reference.)

### 10. qnorm + RoPE + KV insert
- **CUDA**: one fused C++ op
  (`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`,
  `vllm/models/deepseek_v4/attention.py:650`), registered for CUDA only and FP8
  dtype baked at compile-time (corrupts the SWA cache on FNUZ archs).
- **ROCm/gfx12**: always the Python/Triton reference
  `_deepseek_v4_qnorm_rope_kv_insert_reference` (`…/attention.py:141,663`). The
  q-norm + GPT-J RoPE math is torch (`_apply_rope_gptj_last_dims:103`); only the
  K cache write is Triton (`…/common/ops/cache_utils.py:142`). **Partially
  Triton.**

### 16. MoE experts
- The mxfp4 oracle (`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`)
  enumerates AITER backends; the CK variants (`AITER_MXFP4_BF16` W4A16,
  `AITER_MXFP4_MXFP4` W4A4) are gfx950-only. This branch's oracle change (`:252`)
  skips them on gfx12, landing on `AITER_MXFP4_FP8` → `AiterW4A8ExpertsMonolithic`
  (`…/experts/aiter_mxfp4_w4a8_moe.py:166`), whose routing + W4A8 GEMM come from
  `aiter.ops.triton.moe_routing.routing` and `aiter.ops.triton.moe_op_gemm_a8w4`.

### 17. Dense FP8 block-scale linears
- `AiterFp8BlockScaledMMKernel` (`…/scaled_mm/aiter.py:276`) chooses Triton vs CK
  via `self.use_triton` (`:281`). This branch forces it True under
  `VLLM_DSV4_TRITON` so an untuned `(n,k)` never hits the absent CK kernel; the
  Triton kernel is `w8a8_triton_block_scaled_mm`
  (`vllm/model_executor/layers/quantization/utils/fp8_utils.py:841`). The
  selectability flags are widened in `vllm/_aiter_ops.py:1419,1534`.

---

## Remaining non-Triton on gfx12 (open items)

| Op | Current gfx12 backend | Why not Triton yet |
|----|----------------------|--------------------|
| #3 Indexer paged-decode MQA-logits | torch (Python loop) | aiter paged-logits is gluon/CDNA-only; needs a new gfx12 Triton paged kernel or routing to aiter's generic JIT kernel |
| #10 qnorm + RoPE (q side) | torch | fused C++ op is CUDA-only; aiter `fused_reduce_qk_norm_rope_swa_write` contract not yet validated against DSv4 semantics |
| #18 RMSNorm | C++/HIP | works fine; AITER RMSNorm Triton off by default, low value |
| #19 embedding / sampler / all-reduce | native / RCCL | not Triton-portable |
