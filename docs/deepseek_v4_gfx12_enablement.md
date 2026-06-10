# Making DeepSeek-V4-Flash run on gfx12 (Navi48 / RDNA4) — all the tweaks

Short answer: **no, it was not just an env var + a cache fix.** Enabling DSv4 on
gfx12 took changes across **6 categories**: dependency/build, env switches,
Triton kernel wiring, FULL-cudagraph host-sync fixes, third-party package
patches, and disabling gfx9x-only kernels. Full list below.

Final working config:
`VLLM_DSV4_TRITON=1  MOE_BACKEND=triton_unfused  VLLM_ROCM_USE_AITER=1  VLLM_ROCM_USE_AITER_MOE=0  CUDAGRAPH_MODE=FULL_DECODE_ONLY  TP=8`

(AITER is **on** so the sparse-attention indexer uses the upstreamed aiter
`rocm_aiter_sparse_attn_indexer` op; the gfx9x-only aiter sampler is skipped
in-code on gfx12, and MoE stays on `triton_unfused` via `AITER_MOE=0`.)

---

## 1. Dependencies & build (environment)
The checkout targets the torch stable/headeronly ABI (torch 2.11+); the box had
old/mismatched packages, which blocked build and import.

| Package | was | now | why |
|---|---|---|---|
| torch | 2.9.1+rocm7.2 | **2.12.0+rocm7.2** | `_C_stable_libtorch` needs torch≥2.11 stable ABI (`torch/headeronly/...`, `TORCH_BOX`); else build fails |
| torchvision | 0.24.0 | **0.27.0+rocm7.2** | 0.24 vs torch 2.12 → `torchvision::nms` won't register → import crash |
| compressed-tensors | 0.13.0 | **0.17.0** | vLLM pins 0.17 (`compressors.pack_quantized`) |
| kernels (HF) | 0.15.2 | **0.12.3** | transformers 5.9.0 pins `kernels<0.13`; 0.15 crashes at import |
| triton_kernels | AMD `1.0.0+amd` | **OAI `1.0.0`** | AMD variant lacks `matmul_ogs`/`routing`; see §5 |

Also: **rebuilt the vLLM C++/HIP extensions** (`_C`, `_rocm_C`, `_moe_C`,
`_C_stable_libtorch`, …) against torch 2.12.

## 2. Env switches (code)
- **New `VLLM_DSV4_TRITON`** (`vllm/envs.py`): master "all-Triton on gfx12"
  switch — forces aiter-Triton linear/blockscale GEMMs.
- **`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=0`**: AITER on enables the
  aiter sparse-attention indexer op (which already handles the V4 pre-inserted-K
  layout and has non-gfx942 Triton paths); MoE stays on `triton_unfused`.
- The fused Triton sparse-MLA backend (`rocm_aiter_mla_sparse`, upstreamed via
  main) is the only ROCm sparse-MLA path; the qnorm-rope-kv-insert uses main's
  fused `_C` kernel (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`), which
  builds and runs on gfx1201 after rebuilding `_C_stable_libtorch` — `attention.py`
  needs no gfx12 change.
- **RoutedExperts quant fix** (`models/deepseek_v4/quant_config.py`): main's MoE
  refactor (PR #41184) split the MoE into `MoERunner` + `RoutedExperts`, but the
  DeepSeek-V4 quant override only matched `MoERunner`, so the `RoutedExperts`
  weight container fell through to the FP8 method and rejected
  `--moe-backend=triton_unfused`. Fixed to `isinstance(layer, (MoERunner,
  RoutedExperts))` so MXFP4 (`expert_dtype="fp4"`) experts get `Mxfp4MoEMethod`.

## 3. Triton kernel wiring (code)
- **Sparse-MLA attention** uses the upstreamed ROCm impl
  `DeepseekV4ROCMAiterMLAAttention` (`models/deepseek_v4/amd/rocm.py`), selected
  by the AMD model module.
- **aiter linear/GEMM flags honor `VLLM_DSV4_TRITON`**
  (`vllm/_aiter_ops.py:is_linear_enabled / is_triton_gemm_enabled`) — selects
  the aiter-Triton blockscale GEMM **without** flipping global `is_enabled()`.
- **Blockscale GEMM forced to Triton** (`scaled_mm/aiter.py:use_triton`).
- **O-proj wo_a GEMM → `torch.einsum` (rocBLAS)**, same as CDNA. (A Triton
  `batched_gemm_bf16` path was tried but benchmarked ~1.3–1.8× *slower* on
  gfx1201 for this shape — B=n_local_groups=1, K=4096, N=1024 — since rocBLAS
  already works natively, so it was removed.)
- **MHC → aiter Triton**: new adapter `kernels/mhc/aiter_triton.py`
  (`mhc_pre/post/fused_post_pre_aiter_triton` wrapping
  `aiter.ops.triton.fusions.mhc`), wired in `layers/mhc.py` via
  `HAS_AITER_TRITON_MHC`.
- **Indexer → aiter op, non-gluon JIT**: the ROCm `SparseAttnIndexer` dispatches
  to the upstreamed aiter `rocm_aiter_sparse_attn_indexer`
  (`layers/sparse_attn_indexer.py`); `paged_mqa_logits_module()` forces
  `enable_gluon_pa_mqa_logits=False` (gluon is gfx942/gfx950-only) so the decode
  logits land on the portable Triton `deepgemm_fp8_paged_mqa_logits_stage1`.
  (An earlier custom `rocm_sparse_attn_indexer.py` was removed once the aiter op
  was found to handle the V4 pre-inserted-K layout on gfx12.)
- **MoE → `triton_unfused`** (OAI `triton_kernels` MoE, modular so the layer
  does DeepSeek-V4 grouped routing). (Also patched the aiter W4A8 path —
  SILU activation support + `(alpha,limit)` mapping in
  `aiter_mxfp4_w4a8_moe.py` — and added an `EMULATION` weight-conversion branch
  in `oracle/mxfp4.py`; both were stepping stones before `triton_unfused`.)

## 4. FULL-decode cudagraph host-sync fixes (code)  ← "the cache thing"
For `CUDAGRAPH_MODE=FULL_DECODE_ONLY` the decode path must be free of GPU→CPU
syncs and `cudagraph_unsafe` ops:
- **MHC hc_scale cache** (`kernels/mhc/aiter_triton.py:_hc_scale_floats`): the
  aiter `mhc` kernel takes alphas as Python floats → `hc_scale.tolist()` was a
  GPU→CPU sync that aborts capture. Cache the (constant) floats once during
  warmup → sync-free under capture. Dropped its `cudagraph_unsafe` tag.
- The aiter indexer op (`rocm_aiter_sparse_attn_indexer`) is registered as a
  splitting op in `config/compilation.py:_attention_ops`, so it's a graph
  boundary (eager) under PIECEWISE and captured cleanly under FULL_DECODE_ONLY.
- Decouple qnorm from the env so the default Triton path stays self-consistent.

Result: decode TPOT 186 ms (piecewise) → **48 ms** (full), ~3.9×.

## 5. Third-party package patches
- **Installed the OAI `triton_kernels`** (built a wheel from vLLM's vendored
  `vllm/third_party/triton_kernels`) to replace the AMD ROCm variant that lacks
  `matmul_ogs`/`routing` (it's a different API generation).
- **top-k=6 power-of-2 patch** to
  `triton_kernels/routing_details/_routing_compute.py`: pad
  `tl.arange`/`tl.sort` to `next_power_of_2(N_EXPTS_ACT*BLOCK_M)` + mask the
  padding. DeepSeek-V4 top-k=6 → `6*BLOCK_M` isn't a power of 2 (verified
  correct vs torch routing).

## 6. Disabling gfx9x-only kernels + a missing config
- **Sampler → native on gfx12** (`v1/sample/ops/topk_topp_sampler.py`): aiter's
  `top_k_top_p_sampling_from_probs` hardcodes CDNA's 64-lane wavefront
  (`WARP_SIZE=64`, 64-lane shfl masks) → JIT build fails on RDNA4. Since AITER is
  now **on** (for the indexer), the sampler can't be disabled via the global
  flag; instead it's skipped in-code with an `_ON_GFX12X` guard, so gfx12 uses
  vLLM's native sampler while CDNA still gets the aiter one.
- **`gfx1201-MHC_POST.json`** tuning config added (aiter's
  `get_mhc_post_config` has no gfx942 fallback → `FileNotFoundError` otherwise).

---

## TL;DR
| Category | Tweaks |
|---|---|
| Deps/build | torch 2.12, torchvision 0.27, compressed-tensors 0.17, kernels 0.12.3, rebuild `_C*` |
| Env | new `VLLM_DSV4_TRITON`; AITER on + `AITER_MOE=0` |
| Triton wiring | sparse-MLA default, aiter linear/blockscale, MHC, aiter indexer (gluon-off), MoE `triton_unfused` (o-proj GEMM stays rocBLAS einsum) |
| Quant fix | RoutedExperts → MXFP4 method (PR #41184 regression) |
| FULL cudagraph | hc_scale host-sync cache; indexer in `_attention_ops` |
| 3rd-party | OAI `triton_kernels` install + top-k=6 pow2 patch |
| gfx9x-only | native sampler via in-code `_ON_GFX12X` guard; `gfx1201-MHC_POST.json` |

So: the env var + the cache fix were two of many — the bulk was **(a) bringing
the dependency/build stack to torch 2.12**, **(b) routing every CK/ASM/gfx9x
op to a Triton equivalent**, and **(c) the `triton_kernels` install + top-k=6
patch** for the MoE.
