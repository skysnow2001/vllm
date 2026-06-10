# DeepSeek-V4-Flash on gfx12 (Navi48 / RDNA4): Triton wiring changes

Goal: run DeepSeek-V4-Flash on 8× Navi48 (gfx1201) under **FULL_DECODE_ONLY**
cudagraph using **Triton** kernels only — CDNA-only CK / ASM / TileLang /
FlashMLA / DeepGEMM kernels do not build/run on RDNA4.

The Triton kernel selection is gated by **`VLLM_DSV4_TRITON`** (+
`current_platform.is_rocm()`); AITER is also turned **on**
(`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=0`) so the upstreamed aiter
sparse-attention indexer op is reachable, with the gfx9x-only aiter sampler
skipped in-code on gfx12.

The ROCm sparse-MLA *attention backend* and the *sparse indexer*
(`rocm_aiter_mla_sparse`, `rocm_aiter_sparse_attn_indexer`,
`DeepseekV4ROCMAiterMLAAttention`) were upstreamed by `main`; the changes below
are the gfx12 Triton wiring + fixes layered on top.

---

## 1. Selector / env

| File | Change | Why |
|---|---|---|
| `vllm/envs.py` | New `VLLM_DSV4_TRITON` master switch (+15) | Force aiter-Triton variants over CK/ASM, independent of global aiter enablement |
| `vllm/_aiter_ops.py` | `is_linear_enabled()` / `is_triton_gemm_enabled()` also true under switch; `if_aiter_supported`→True; gluon paths forced off; fp8 `paged_mqa_logits` loader gated on switch (+13) | Make aiter Triton ops selectable on gfx12; gluon kernels don't run on RDNA4 |
| `scripts/serve_deepseek_v4_flash.sh` | `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=0` | AITER on → aiter sparse-attention indexer usable; MoE stays `triton_unfused` |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | `_ON_GFX12X` guard → native sampler on gfx12 even with AITER on | aiter sampler is gfx9x-only (`WARP_SIZE=64`) |

## 2. Sparse MLA attention

| File | Change | Why |
|---|---|---|
| `vllm/model_executor/layers/sparse_attn_indexer.py` | ROCm `SparseAttnIndexer` dispatches to the aiter `rocm_aiter_sparse_attn_indexer` op | aiter op handles V4 pre-inserted-K (`skip_k_cache_insert`/`k=None`) + non-gfx942 Triton |
| `vllm/_aiter_ops.py` (`paged_mqa_logits_module`) | gluon dispatch forced off → portable Triton `deepgemm_fp8_paged_mqa_logits_stage1` | aiter's default gluon `pa_mqa_logits` is gfx942/gfx950-only |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | o-proj `rocm_inv_rope_einsum` keeps `torch.einsum` (rocBLAS) — an aiter `batched_gemm_bf16` Triton path was tried then removed (benchmarked ~1.3–1.8× slower on gfx1201 at B=1, K=4096, N=1024) | rocBLAS works natively on ROCm; no Triton replacement needed (same as CDNA) |
| qnorm+RoPE+KV-insert (`attention.py`) | uses main's fused `_C` kernel `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` — **no gfx12 code** (`attention.py` identical to main) | the `_C` kernel is 32-lane + OCP-fp8 safe; just rebuilt into `_C_stable_libtorch` for gfx1201 (only the build arch list was needed) |

## 3. MHC (hyper-connection)

| File | Change | Why |
|---|---|---|
| `vllm/model_executor/kernels/mhc/aiter_triton.py` | **New** adapters to aiter Triton `fusions.mhc.{mhc, mhc_post, mhc_post_pre}` (+278) | Triton MHC for gfx12 (PR #41946 kernels) |
| `vllm/model_executor/kernels/mhc/__init__.py` | Expose `has_aiter_triton_mhc()` (+5) | Capability probe |
| `vllm/model_executor/layers/mhc.py` | Dispatch to Triton MHC on gfx12 + CPU-side `hc_scale`-floats cache keyed by `(data_ptr, numel)` (+100) | Avoid a GPU→CPU host-sync that aborts cudagraph capture |
| `aiter_scripts/aiter/ops/triton/configs/gfx1201-MHC_POST.json` | Tuning config (+9) | `mhc_post` has no gfx942 fallback config |

## 4. Linear / GEMM

| File | Change | Why |
|---|---|---|
| `vllm/model_executor/kernels/linear/scaled_mm/aiter.py` | `use_triton` true under switch (non-FNUZ) → aiter Triton W8A8 (+9) | CK / hipBLASLt W8A8 path not usable on gfx12 |

## 5. MoE

| File | Change | Why |
|---|---|---|
| `aiter_scripts/vllm/third_party/triton_kernels/routing_details/_routing_compute.py` | top-k=6 power-of-2 fix for the OAI `triton_kernels` router (+158) | `tl.arange(0, N_EXPTS_ACT*BLOCK_M)` asserts on top-k=6 (live copy is under gitignored `vllm/third_party/`) |

MoE backend used: **`triton_unfused`** → `UnfusedOAITritonExperts` (OpenAI
`triton_kernels` modular path). No vLLM source change is required for it beyond
the router pow2 patch above. The earlier `aiter` (AITER_MXFP4_FP8) and
`emulation` MoE experiments were removed (see §7).

## 6. Model loading / merge-glue (minimal)

| File | Change | Why |
|---|---|---|
| `vllm/models/deepseek_v4/quant_config.py` | `_resolve_deepseek_v4_expert_dtype` infers expert layout when `expert_dtype` missing (+47); **`get_quant_method`/`is_mxfp4_quant` match `(MoERunner, RoutedExperts)`** | (1) DSv4-Flash FP8 checkpoint has no explicit `expert_dtype`; (2) PR #41184 regression — `RoutedExperts` (the actual `get_quant_method` caller) missed the `MoERunner`-only check → MXFP4 experts fell through to FP8 and rejected `triton_unfused` |
| `vllm/models/deepseek_v4/amd/model.py` | Import fix after main's refactor (+5) | Use main's `DeepseekV4ROCMAiterMLAAttention` + `_resolve_deepseek_v4_expert_dtype` |
| `vllm/models/deepseek_v4/nvidia/model.py` | Import fix (+7) | main-refactor resolution |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | `not is_rocm()` guard on a tile-metadata assert (+18) | Shared FlashMLA metadata is subclassed on ROCm |

---

## 7. Removed during cleanup (after merging main)

| Removed | Why now unnecessary |
|---|---|
| `vllm/v1/attention/ops/rocm_flash_mla_sparse.py` (−651) + its `ops/flashmla.py` ROCm elif | main upstreamed `rocm_sparse_attn_decode/prefill`; the FlashMLA torch fallback is never called on ROCm (`get_mla_metadata` is skipped for `is_rocm()`). ROCm falls through to the existing raising-stub `else` branch. |
| `VLLM_ROCM_USE_V4_TRITON_FALLBACK` env | Its only consumer was the runtime selector main removed + the deleted fallback file; toggled nothing. |
| aiter `dsv4_qnorm_rope_kv_insert.py` copies | Never wired — the fused `_C` kernel (rebuilt for gfx1201) is used instead. |
| qnorm torch reference (`_deepseek_v4_qnorm_rope_kv_insert_reference`, `_apply_rope_gptj_last_dims`) | Removed once the fused `_C` kernel was confirmed working on gfx12 — `attention.py` is now identical to main. |
| `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py` (−581) + its `_attention_ops` entry + `sparse_attn_indexer.py` custom branch | The upstreamed aiter `rocm_aiter_sparse_attn_indexer` op handles the V4 pre-inserted-K layout on gfx12 (gluon off); validated at 95% GSM8K. AITER turned on (with the in-code gfx12 sampler guard) so the aiter indexer op is reachable. |
| `oracle/mxfp4.py` EMULATION + `aiter` (NAVI48-TEST) hunks; `aiter_mxfp4_w4a8_moe.py` activation fix | Both MoE experiments abandoned in favor of `triton_unfused`; reverted to main. |
| `emulation` / `aiter` options in `serve_deepseek_v4_flash.sh` | MoE backends are now `auto \| triton \| triton_unfused`. |

## 8. Supporting (non-code)

`docs/` (gfx12 enablement, kernel matrix, full-vs-piecewise bench), `results/*.json`
(bench runs), `scripts/` (serve / bench / sanity / clean / run_dsv4),
`aiter_scripts/README.md` + the snapshots above.

## Result

FULL_DECODE_ONLY + Triton ≈ **3.9× faster** than the old piecewise + torch path
(47.6 ms vs 186 ms TPOT @ conc=1, 1024/1024).
