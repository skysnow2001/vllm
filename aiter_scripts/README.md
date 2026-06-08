# aiter gfx1201 (Navi48) additions for DeepSeek-V4

New files added to the aiter checkout to run DeepSeek-V4-Flash on gfx12.
Paths below are relative to the aiter repo root.

- `aiter/ops/triton/configs/gfx1201-MHC_POST.json`
    Tuning config for the MHC `mhc_post`/`mhc_post_pre` Triton kernels.
    `get_mhc_post_config` has NO gfx942 fallback (unlike MHC_FUSED), so a
    `gfx1201-MHC_POST.json` must exist or it raises FileNotFoundError.
    (Temporary: mirrors gfx942 block sizes; not perf-tuned for RDNA4.)

- `aiter/ops/triton/fusions/dsv4_qnorm_rope_kv_insert.py` (launcher)
- `aiter/ops/triton/_triton_kernels/fusions/dsv4_qnorm_rope_kv_insert.py` (kernel)
    Editable copy of `fused_reduce_qk_norm_rope_swa_write` for adapting the
    DSv4 qnorm+RoPE+KV-insert (#10) to a Triton kernel. NOT yet wired.

NOTE: the top-k=6 pow2 fix to `triton_kernels/routing_details/_routing_compute.py`
lives in the vLLM repo at `vllm/third_party/triton_kernels/...` (tracked there),
not in this folder.

- `vllm/third_party/triton_kernels/routing_details/_routing_compute.py`
    top-k=6 power-of-2 fix for the OAI triton_kernels routing kernel
    (pad tl.arange/tl.sort to next_power_of_2 + mask). Lives under
    vllm/third_party/ which is gitignored, so it's copied here to preserve it.
