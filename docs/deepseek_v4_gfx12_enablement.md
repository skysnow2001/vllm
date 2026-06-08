# Making DeepSeek-V4-Flash run on gfx12 (Navi48 / RDNA4) — all the tweaks

Short answer: **no, it was not just an env var + a cache fix.** Enabling DSv4 on
gfx12 took changes across **6 categories**: dependency/build, env switches,
Triton kernel wiring, FULL-cudagraph host-sync fixes, third-party package
patches, and disabling gfx9x-only kernels. Full list below.

Final working config:
`VLLM_DSV4_TRITON=1  MOE_BACKEND=triton_unfused  VLLM_ROCM_USE_AITER=0  CUDAGRAPH_MODE=FULL_DECODE_ONLY  TP=8`

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
  switch — forces aiter-Triton linear/blockscale + o-proj, keeps the Triton
  sparse indexer, decouples qnorm from the CK path.
- **Flipped `VLLM_ROCM_USE_V4_TRITON_FALLBACK` default → False**
  (`vllm/envs.py`): Triton sparse-MLA is now the ROCm default; the torch
  online-softmax path is opt-in backup.

## 3. Triton kernel wiring (code)
- **Sparse-MLA attention** is the default ROCm impl
  (`models/deepseek_v4/attention.py:_select_v4_sparse_impl`).
- **aiter linear/GEMM flags honor `VLLM_DSV4_TRITON`**
  (`vllm/_aiter_ops.py:is_linear_enabled / is_triton_gemm_enabled`) — selects
  the aiter-Triton blockscale GEMM **without** flipping global `is_enabled()`.
- **Blockscale GEMM forced to Triton** (`scaled_mm/aiter.py:use_triton`).
- **O-proj grouped GEMM → Triton**: `rocm_aiter_mla_sparse.py:
  _rocm_triton_grouped_oproj_gemm` (aiter `batched_gemm_bf16`, with an explicit
  launch config since aiter ships no `gfx1201-BATCHED_GEMM-A16W16` json).
- **MHC → aiter Triton**: new adapter `kernels/mhc/aiter_triton.py`
  (`mhc_pre/post/fused_post_pre_aiter_triton` wrapping
  `aiter.ops.triton.fusions.mhc`), wired in `layers/mhc.py` via
  `HAS_AITER_TRITON_MHC`.
- **Indexer decode → non-gluon JIT**: `paged_mqa_logits_module()` forces
  `enable_gluon_pa_mqa_logits=False` (gluon is gfx950/gfx1250-only); the Triton
  no-insert indexer is the default ROCm path
  (`layers/sparse_attn_indexer.py`).
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
- **Indexer op tag made conditional** (`rocm_sparse_attn_indexer.py`): drop
  `cudagraph_unsafe` under `VLLM_DSV4_TRITON` (decode path is the sync-free
  Triton kernel; prefill runs eager under FULL_DECODE_ONLY).
- Decouple qnorm/indexer from the env so the default Triton path stays
  self-consistent.

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
- **Sampler → native** (`VLLM_ROCM_USE_AITER=0` for triton_unfused in
  `serve_deepseek_v4_flash.sh`): aiter's `top_k_top_p_sampling_from_probs`
  hardcodes CDNA's 64-lane wavefront (`WARP_SIZE=64`, 64-lane shfl masks) → JIT
  build fails on RDNA4. vLLM's native sampler is correct on any arch.
- **`gfx1201-MHC_POST.json`** tuning config added (aiter's
  `get_mhc_post_config` has no gfx942 fallback → `FileNotFoundError` otherwise).

---

## TL;DR
| Category | Tweaks |
|---|---|
| Deps/build | torch 2.12, torchvision 0.27, compressed-tensors 0.17, kernels 0.12.3, rebuild `_C*` |
| Env | new `VLLM_DSV4_TRITON`; flip `VLLM_ROCM_USE_V4_TRITON_FALLBACK` default |
| Triton wiring | sparse-MLA default, aiter linear/blockscale, o-proj GEMM, MHC, indexer (gluon-off), MoE `triton_unfused` |
| FULL cudagraph | hc_scale host-sync cache; conditional `cudagraph_unsafe` tag |
| 3rd-party | OAI `triton_kernels` install + top-k=6 pow2 patch |
| gfx9x-only | native sampler (disable aiter sampler); `gfx1201-MHC_POST.json` |

So: the env var + the cache fix were two of many — the bulk was **(a) bringing
the dependency/build stack to torch 2.12**, **(b) routing every CK/ASM/gfx9x
op to a Triton equivalent**, and **(c) the `triton_kernels` install + top-k=6
patch** for the MoE.
