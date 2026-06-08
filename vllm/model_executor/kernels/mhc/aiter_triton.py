# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER **Triton** mHC kernels for DeepSeek-V4 on ROCm (e.g. gfx12/Navi48).

These wrap the fused Triton kernels in ``aiter.ops.triton.fusions.mhc``
(``mhc`` / ``mhc_post`` / ``mhc_post_pre``) — distinct from the CK kernels in
``aiter.ops.mhc`` used by ``mhc_pre_aiter`` / ``mhc_post_aiter``. They are the
default MHC path on ROCm; ``mhc_*_torch`` remains the backup when the aiter
Triton module is unavailable.

Adapter notes (vLLM convention -> aiter convention):
  * ``fn`` (N=hc_mult3, K=hc_mult*hidden) fp32  ->  ``phi`` = fn.T  (K, N) bf16
  * ``hc_scale`` (3,) fp32                       ->  alpha_pre/post/res / alphas
  * ``hc_base`` (hc_mult3,) fp32                 ->  ``bias`` (N,) fp32
  * ``sinkhorn_repeat``                          ->  ``sinkhorn_iters``
  * residual (M, hc_mult, hidden)                ->  x/residual reshaped (M, ...)

Numerics differ slightly from ``mhc_*_torch`` (aiter folds the RMS scale +
bias into the projection and uses a log-domain Sinkhorn); this is intentional
per the current "wire it, don't chase the error" goal.
"""
from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def has_aiter_triton_mhc() -> bool:
    """True when aiter's fused Triton mHC kernels are importable."""
    try:
        from aiter.ops.triton.fusions import mhc as _m  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# mHC pre  (aiter.ops.triton.fusions.mhc.mhc)
# ---------------------------------------------------------------------------
# Cache hc_scale (a constant param) -> python float alphas, keyed by storage
# pointer, so the GPU->CPU read happens once during eager warmup and never
# inside a captured cudagraph (where a sync would abort capture).
_HC_SCALE_FLOATS_CACHE: dict[tuple[int, int], tuple[float, float, float]] = {}


def _hc_scale_floats(hc_scale: torch.Tensor) -> tuple[float, float, float]:
    key = (hc_scale.data_ptr(), hc_scale.numel())
    vals = _HC_SCALE_FLOATS_CACHE.get(key)
    if vals is None:
        a_pre, a_post, a_res = (float(v) for v in hc_scale.tolist())
        vals = (a_pre, a_post, a_res)
        _HC_SCALE_FLOATS_CACHE[key] = vals
    return vals


def mhc_pre_aiter_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fusions.mhc import mhc as aiter_mhc

    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    M = residual.numel() // (hc_mult * hidden)

    x = residual.reshape(M, hc_mult * hidden)
    phi = fn.t().to(residual.dtype).contiguous()  # (K, N)
    # aiter's `mhc` takes per-stream alphas as python floats. Reading them off
    # the device is a GPU->CPU sync that would abort a FULL cudagraph capture,
    # so cache by tensor identity: hc_scale is a constant parameter, populated
    # during eager warmup before capture, then reused sync-free under capture.
    a_pre, a_post, a_res = _hc_scale_floats(hc_scale)

    h_post, h_res, layer_input = aiter_mhc(
        x,
        phi,
        a_pre,
        a_post,
        a_res,
        hc_base,
        hc_mult,
        eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
    )
    post_mix = h_post.to(torch.float32).reshape(*outer, hc_mult, 1)
    comb_mix = h_res.to(torch.float32).reshape(*outer, hc_mult, hc_mult)
    layer_input = layer_input.reshape(*outer, hidden)
    return post_mix, comb_mix, layer_input


def _mhc_pre_aiter_triton_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    post_mix = torch.empty(
        *outer, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        *outer, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        *outer, hidden, dtype=torch.bfloat16, device=residual.device
    )
    return post_mix, comb_mix, layer_input


# ---------------------------------------------------------------------------
# mHC post  (aiter.ops.triton.fusions.mhc.mhc_post)
# ---------------------------------------------------------------------------
def mhc_post_aiter_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from aiter.ops.triton.fusions.mhc import mhc_post as aiter_mhc_post

    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    M = residual.numel() // (hc_mult * hidden)

    layer_input = x.reshape(M, hidden)
    res3 = residual.reshape(M, hc_mult, hidden)
    plm = post_layer_mix.reshape(M, hc_mult).to(torch.float32)
    comb = comb_res_mix.reshape(M, hc_mult, hc_mult).to(torch.float32)

    out = aiter_mhc_post(None, layer_input, res3, plm, comb)
    return out.reshape(*outer, hc_mult, hidden)


def _mhc_post_aiter_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


# ---------------------------------------------------------------------------
# mHC fused post + (next) pre  (aiter.ops.triton.fusions.mhc.mhc_post_pre)
# ---------------------------------------------------------------------------
def mhc_fused_post_pre_aiter_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fusions.mhc import mhc_post_pre as aiter_mhc_post_pre

    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    M = residual.numel() // (hc_mult * hidden)

    layer_input = x.reshape(M, hidden)
    res3 = residual.reshape(M, hc_mult, hidden)
    plm = post_layer_mix.reshape(M, hc_mult).to(torch.float32)
    comb = comb_res_mix.reshape(M, hc_mult, hc_mult).to(torch.float32)
    phi = fn.t().to(residual.dtype).contiguous()  # (K, N)
    alphas = hc_scale.to(torch.float32)  # (3,) tensor — no host read

    h_post, h_res, layer_input_out, residual_out = aiter_mhc_post_pre(
        layer_input,
        res3,
        plm,
        comb,
        phi,
        alphas,
        hc_base,
        hc_mult,
        eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
    )
    # vLLM contract: (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur)
    return (
        residual_out.reshape(*outer, hc_mult, hidden),
        h_post.to(torch.float32).reshape(*outer, hc_mult, 1),
        h_res.to(torch.float32).reshape(*outer, hc_mult, hc_mult),
        layer_input_out.reshape(*outer, hidden),
    )


def _mhc_fused_post_pre_aiter_triton_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    residual_out = torch.empty_like(residual)
    post_mix = torch.empty(
        *outer, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        *outer, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        *outer, hidden, dtype=torch.bfloat16, device=residual.device
    )
    return residual_out, post_mix, comb_mix, layer_input


# mhc_pre's only host read (hc_scale -> floats) is cached in _hc_scale_floats,
# so under capture it's sync-free and cudagraph-safe (no tag). mhc_post /
# mhc_post_pre keep everything on-device (alphas stays a tensor) too.
direct_register_custom_op(
    op_name="mhc_pre_aiter_triton",
    op_func=mhc_pre_aiter_triton,
    mutates_args=[],
    fake_impl=_mhc_pre_aiter_triton_fake,
)
direct_register_custom_op(
    op_name="mhc_post_aiter_triton",
    op_func=mhc_post_aiter_triton,
    mutates_args=[],
    fake_impl=_mhc_post_aiter_triton_fake,
)
direct_register_custom_op(
    op_name="mhc_fused_post_pre_aiter_triton",
    op_func=mhc_fused_post_pre_aiter_triton,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_aiter_triton_fake,
)
