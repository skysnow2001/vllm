# DeepSeek-V4-Flash on gfx12 (Navi48 / RDNA4): Triton wiring changes

Goal: run DeepSeek-V4-Flash on 8× Navi48 (gfx1201) under **FULL_DECODE_ONLY**
cudagraph using **Triton** kernels only — CDNA-only CK / ASM / TileLang /
FlashMLA / DeepGEMM kernels do not build/run on RDNA4.

Everything is gated by one master switch, **`VLLM_DSV4_TRITON`** (+
`current_platform.is_rocm()`). With the switch off, behavior is identical to
upstream `main`. Net code delta vs `origin/main`: **15 files, ~1264 / −28**.

The ROCm sparse-MLA *attention backend* itself (`rocm_sparse_attn_decode` /
`rocm_sparse_attn_prefill`, `DeepseekV4ROCMAiterMLAAttention`) was upstreamed by
`main`; the changes below are the gfx12 Triton wiring layered on top.

---

## 1. Selector / env

| File | Change | Why |
|---|---|---|
| `vllm/envs.py` | New `VLLM_DSV4_TRITON` master switch (+15) | Force aiter-Triton variants over CK/ASM, independent of global aiter enablement |
| `vllm/_aiter_ops.py` | `is_linear_enabled()` / `is_triton_gemm_enabled()` also true under switch; `if_aiter_supported`→True; gluon paths forced off; fp8 `paged_mqa_logits` loader gated on switch (+13) | Make aiter Triton ops selectable on gfx12; gluon kernels don't run on RDNA4 |
| `vllm/config/compilation.py` | cudagraph tag tweak (+1) | Allow FULL_DECODE_ONLY capture of the sparse path |

## 2. Sparse MLA attention

| File | Change | Why |
|---|---|---|
| `vllm/v1/attention/ops/rocm_sparse_attn_indexer.py` | **New** Triton fp8 paged MQA-logits sparse indexer (+581) | gfx12 indexer (no CK/gluon) |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | Dispatch ROCm `SparseAttnIndexer` to the Triton path (+26) | Default when AITER off / forced by switch |
| `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | `rocm_inv_rope_einsum` o-proj uses an explicit-config aiter `batched_gemm_bf16` Triton GEMM under switch, torch.einsum fallback (+61) | No gfx1201 BATCHED_GEMM tuning json exists |
| `vllm/models/deepseek_v4/attention.py` | qnorm+RoPE+KV-insert torch/Triton reference (`_deepseek_v4_qnorm_rope_kv_insert_reference`, `_apply_rope_gptj_last_dims`) used on ROCm (+126) | Fused `_C` op is CUDA-only and its FP8 type (OCP vs FNUZ) is wrong on ROCm |

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
| `vllm/models/deepseek_v4/quant_config.py` | `_resolve_deepseek_v4_expert_dtype` infers expert layout when `expert_dtype` missing (+47) | DSv4-Flash FP8 checkpoint has no explicit `expert_dtype` |
| `vllm/models/deepseek_v4/amd/model.py` | Import fix after main's refactor (+5) | Use main's `DeepseekV4ROCMAiterMLAAttention` + `_resolve_deepseek_v4_expert_dtype` |
| `vllm/models/deepseek_v4/nvidia/model.py` | Import fix (+7) | main-refactor resolution |
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | `not is_rocm()` guard on a tile-metadata assert (+18) | Shared FlashMLA metadata is subclassed on ROCm |

---

## 7. Removed during cleanup (after merging main)

| Removed | Why now unnecessary |
|---|---|
| `vllm/v1/attention/ops/rocm_flash_mla_sparse.py` (−651) + its `ops/flashmla.py` ROCm elif | main upstreamed `rocm_sparse_attn_decode/prefill`; the FlashMLA torch fallback is never called on ROCm (`get_mla_metadata` is skipped for `is_rocm()`). ROCm falls through to the existing raising-stub `else` branch. |
| `VLLM_ROCM_USE_V4_TRITON_FALLBACK` env | Its only consumer was the runtime selector main removed + the deleted fallback file; toggled nothing. |
| aiter `dsv4_qnorm_rope_kv_insert.py` copies | Never wired — the torch reference is used instead. |
| `oracle/mxfp4.py` EMULATION + `aiter` (NAVI48-TEST) hunks; `aiter_mxfp4_w4a8_moe.py` activation fix | Both MoE experiments abandoned in favor of `triton_unfused`; reverted to main. |
| `emulation` / `aiter` options in `serve_deepseek_v4_flash.sh` | MoE backends are now `auto \| triton \| triton_unfused`. |

## 8. Supporting (non-code)

`docs/` (gfx12 enablement, kernel matrix, full-vs-piecewise bench), `results/*.json`
(bench runs), `scripts/` (serve / bench / sanity / clean / run_dsv4),
`aiter_scripts/README.md` + the snapshots above.

## Result

FULL_DECODE_ONLY + Triton ≈ **3.9× faster** than the old piecewise + torch path
(47.6 ms vs 186 ms TPOT @ conc=1, 1024/1024).
