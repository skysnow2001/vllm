# DeepSeek-V4-Flash kernel matrix — CUDA vs CDNA(MI3xx) vs gfx12 (Navi48, current)

Per-operation backend for DeepSeek-V4-Flash. Column 4 is what this branch
**currently runs on gfx12/RDNA4** (config: `VLLM_DSV4_TRITON=1`,
`MOE_BACKEND=triton_unfused`, `VLLM_ROCM_USE_AITER=0`,
`CUDAGRAPH_MODE=FULL_DECODE_ONLY`). Paths are `file:symbol`.

Type legend — **CuteDSL / CUTLASS / DeepGEMM / FlashMLA / FlashInfer / TRT-LLM** =
NVIDIA libs; **CK / ASM** = AITER composable-kernel / assembly (CDNA, mostly
gfx942/gfx950); **TileLang** = tilelang codegen (no HIP backend); **Triton** =
Triton JIT; **C++/HIP** = compiled `_C` op; **torch** = eager reference.

| Operation | CUDA (NVIDIA) | CDNA / MI3xx (gfx942/950) | **gfx12 current (Triton)** |
|---|---|---|---|
| MLA sparse attention (prefill+decode) | **FlashMLA** `nvidia/flashmla.py:flash_mla_sparse_fwd / flash_mla_with_kvcache` (CUDA C++) | Triton `rocm_aiter_mla_sparse.py` (no FlashMLA on ROCm) | **Triton** `v1/attention/ops/rocm_aiter_mla_sparse.py:rocm_sparse_attn_prefill / rocm_sparse_attn_decode` (in-kernel FP8 dequant + online softmax + attn_sink) |
| Indexer — prefill MQA-logits | **DeepGEMM** `sparse_attn_indexer.py:233:fp8_fp4_mqa_logits` | AITER `fp8_mqa_logits` (gluon/CK) | **Triton** `rocm_sparse_attn_indexer.py:149:_mqa_logits_triton` |
| Indexer — decode paged MQA-logits | **DeepGEMM** `sparse_attn_indexer.py:314:fp8_fp4_paged_mqa_logits` | AITER gluon `pa_mqa_logits` (gfx942/950) | **Triton** aiter `pa_mqa_logits.deepgemm_fp8_paged_mqa_logits_stage1` (non-gluon JIT) via `rocm_aiter_mla_sparse.py:rocm_fp8_paged_mqa_logits` |
| Indexer — top-k select | C++ `_C.top_k_per_row_{prefill,decode}` | C++/HIP `_C.top_k_per_row_*` | **C++/HIP** `_C.top_k_per_row_*` (compiled op, not Triton) |
| KV compressor (compress→norm→RoPE→quant→store) | **CuteDSL** `nvidia/ops/sparse_attn_compress_cutedsl.py:compress_norm_rope_store_cutedsl` (head_dim=512 only) | **Triton** `common/ops/fused_compress_quant_cache.py:compress_norm_rope_store_triton` (same kernel as gfx12) | **Triton** `common/ops/fused_compress_quant_cache.py:compress_norm_rope_store_triton` (+ `save_partial_states`) |
| MHC pre | **TileLang** `mhc_pre_tilelang` | CK `aiter.ops.mhc` (HIP, disabled — accuracy) | **Triton** aiter `aiter.ops.triton.fusions.mhc.mhc` via `kernels/mhc/aiter_triton.py:mhc_pre_aiter_triton` |
| MHC post | **TileLang** `mhc_post_tilelang` | CK `aiter.ops.mhc` (disabled) | **Triton** aiter `…fusions.mhc.mhc_post` via `aiter_triton.py:mhc_post_aiter_triton` |
| MHC fused post+pre | **TileLang** `mhc_fused_post_pre_tilelang` | torch compose | **Triton** aiter `…fusions.mhc.mhc_post_pre` via `aiter_triton.py:mhc_fused_post_pre_aiter_triton` (needs `gfx1201-MHC_POST.json`) |
| HC head (MTP vocab gate) | **TileLang** `hc_head_fused_kernel_tilelang` | Triton `hc_head_triton` | **Triton** `kernels/mhc/triton.py:hc_head_triton` |
| O-proj — inverse RoPE | Triton (`fused_inv_rope_fp8_quant`) | torch | **torch** ref `rocm_aiter_mla_sparse.py:_apply_inv_rope_ref` (`rotary_emb.forward_native`) |
| O-proj — wo_a grouped GEMM | **DeepGEMM** `fp8_einsum` | torch `torch.einsum` | **Triton** aiter `gemm/batched/batched_gemm_bf16` via `rocm_aiter_mla_sparse.py:_rocm_triton_grouped_oproj_gemm` |
| qnorm + RoPE (q side) | C++ `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | C++ (CUDA-only) → torch | **torch** ref `attention.py:_apply_rope_gptj_last_dims` |
| KV FP8 quant + cache insert | (part of the C++ fused op) | torch + Triton | **Triton** `common/ops/cache_utils.py:quantize_and_insert_k_cache` |
| Indexer Q proj + RoPE + FP8 quant | Triton `fused_indexer_q_rope_quant` | Triton | **Triton** `common/ops/fused_indexer_q.py:fused_indexer_q_rope_quant` |
| Q/KV RMSNorm (fused, attn) | Triton `fused_q_kv_rmsnorm` | Triton | **Triton** `common/ops/fused_qk_rmsnorm.py:fused_q_kv_rmsnorm` |
| MTP input / shared-head RMSNorm | Triton | Triton | **Triton** `common/ops/fused_mtp_input_rmsnorm.py` |
| KV-cache dequant + gather (prefill pool) | Triton | Triton | **Triton** `common/ops/cache_utils.py:dequantize_and_gather_k_cache` |
| Top-k / SWA index plumbing | n/a (FlashMLA does it internally) | **Triton** `models/deepseek_v4/amd/rocm.py` (same kernel as gfx12) | **Triton** `models/deepseek_v4/amd/rocm.py` (combine/ragged/global-topk) |
| MoE experts (routing + grouped W4 GEMM, MXFP4) | **DeepGEMM/TRT-LLM/FlashInfer FP4** `oracle/mxfp4.py` (`DeepGemmFP4Experts` / `TrtLlmMxfp4Experts` / `FlashInferExperts`) | **CK** `AiterExperts` (W4A16/W4A4, gfx950-only) | **Triton** OAI `experts/gpt_oss_triton_kernels_moe.py:UnfusedOAITritonExperts` → `triton_kernels.matmul_ogs` (+ top-k=6 routing pow2 patch); layer does DeepSeek-V4 grouped routing |
| Dense FP8 block-scale linears (fused_wqa_wkv, wq_b, wo_b, gate) | **FlashInfer/DeepGEMM/CUTLASS** block-scaled mm (`linear/__init__.py` CUDA list) | **CK** aiter `gemm_a8w8_blockscale` | **Triton** aiter `triton_gemm_a8w8_blockscale` (`scaled_mm/aiter.py:AiterFp8BlockScaledMMKernel`, `use_triton` forced) → `fp8_utils.py:w8a8_triton_block_scaled_mm` available too |
| RMSNorm (generic transformer norm) | C++ `_C.rms_norm` | C++/HIP `_C.rms_norm` or AITER | **C++/HIP** `_C.rms_norm` (`layers/layernorm.py`; not Triton, aiter-rmsnorm off) |
| Sampler (top-k / top-p) | **FlashInfer** `flashinfer.sampling` | **CK/HIP** aiter `top_k_top_p_sampling_from_probs` (64-lane wavefront, gfx9x-only) | **native** torch/Triton `v1/sample/ops/topk_topp_sampler.py:forward_native` (aiter one is gfx9x-only) |
| Token embedding / sampler glue / TP all-reduce | native / RCCL | native / RCCL | **native / RCCL** (not Triton-portable) |

## Summary for gfx12 (current)
**Triton today:** MLA sparse attention, indexer prefill+decode MQA-logits, KV
compressor, MHC pre/post/fused, HC head, o-proj grouped GEMM, indexer-Q,
q/kv-norm, MTP norms, KV dequant/gather, index plumbing, **MoE experts
(triton_unfused / matmul_ogs)**, **dense FP8 block-scale linears**.

**Still not Triton on gfx12** (work fine, just not Triton):
- qnorm + q-side RoPE → **torch** ref; o-proj inverse-RoPE → **torch** (`forward_native`)
- KV insert / indexer top-k → **C++/HIP `_C`** ops
- generic RMSNorm → **C++/HIP `_C.rms_norm`**
- sampler → **native** (aiter sampler is gfx9x-only: hardcoded WARP_SIZE=64)
- embedding / TP all-reduce → native / RCCL

## gfx12 enablement notes (what made the Triton paths work)
- `VLLM_DSV4_TRITON=1`: forces aiter-Triton blockscale GEMM + o-proj, keeps the
  Triton sparse indexer, decouples qnorm/indexer from the CK paths.
- `MOE_BACKEND=triton_unfused`: OAI `triton_kernels` MoE (modular → DeepSeek-V4
  routing in the layer). Requires the **OAI `triton_kernels`** package (the AMD
  ROCm variant lacks `matmul_ogs`/`routing`) and a **top-k=6 pow2 patch** to
  `routing_details/_routing_compute.py`.
- `VLLM_ROCM_USE_AITER=0`: avoids the gfx9x-only aiter sampler → native sampler.
- `gfx1201-MHC_POST.json`: required MHC_POST tuning config (no gfx942 fallback).
