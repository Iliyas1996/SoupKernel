"""Our own fused RMSNorm (Triton), forward and backward, for Qwen2RMSNorm, plus a fused
"residual add + RMSNorm" for the norm that follows each attention block.

Why: Qwen2 runs RMSNorm 57 times per forward pass (2 per decoder layer + the final norm). The
PyTorch version is about ten small kernels each way (cast to fp32, pow, mean, add, rsqrt, mul,
cast, mul, and their gradients), and autograd keeps several full-size fp32 copies of the
activations for the backward pass. Here it is one launch forward and one launch backward, and
only the input and one fp32 number per row are kept for the backward pass.

What it computes, exactly as transformers' Qwen2RMSNorm.forward:
    n   = x_fp32 * rsqrt(mean(x_fp32 ** 2) + eps)       # in fp32
    out = weight * n.to(x.dtype)                         # dtype = promote(weight, x)
Backward (for the input; the norm weights are frozen under LoRA):
    dn  = (dout * weight), rounded like PyTorch's autograd does (to out dtype, then x dtype)
    dx  = rstd * dn - x * rstd**3 * mean(dn * x)

Fused add (ADD=True): in each decoder layer transformers computes
    h = residual + attention_output;   x = post_attention_layernorm(h)
and keeps h as the next residual. The forward kernel reads both inputs once and writes h and x;
the backward kernel adds the gradient arriving at h (from the residual path) to the norm's input
gradient, which replaces a separate gradient-accumulation add.

The math is the standard RMSNorm derivation (Zhang & Sennrich 2019, "Root Mean Square Layer
Normalization"); fusing it into one kernel per direction, and fusing the residual add into the
norm, are common ideas (for example the Triton layer-norm tutorial and the "add + RMSNorm" fusions
in inference engines); the kernel code here is our own.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, delta_ptr, sum_ptr, w_ptr, out_ptr, rstd_ptr,
                       x_row_stride, delta_row_stride, sum_row_stride, out_row_stride, n_cols, eps,
                       ADD: tl.constexpr, BLOCK: tl.constexpr):
    """One program normalizes one row (one token). With ADD, the row is first x + delta, which is
    also written out (it is the next residual). `sum_ptr` has the norm input's dtype."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
    if ADD:
        delta = tl.load(delta_ptr + row * delta_row_stride + cols, mask=inside,
                        other=0.0).to(tl.float32)
        # round the sum to its dtype, as `residual + hidden_states` does, then normalize that
        h = (x + delta).to(sum_ptr.dtype.element_ty)
        tl.store(sum_ptr + row * sum_row_stride + cols, h, mask=inside)
        x = h.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(rstd_ptr + row, rstd)
    # `.to(input_dtype)` in transformers rounds the normalized value to the norm's input dtype
    normed = (x * rstd).to(sum_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, (w * normed).to(out_ptr.dtype.element_ty),
             mask=inside)


@triton.jit
def rmsnorm_bwd_kernel(dout_ptr, dsum_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
                       dout_row_stride, dsum_row_stride, x_row_stride, dx_row_stride, n_cols,
                       ADD_DSUM: tl.constexpr, BLOCK: tl.constexpr):
    """One program computes the input gradient of one row. `x` is the norm's input (the sum h in
    the fused case); with ADD_DSUM the gradient that reached h directly is added."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    dout = tl.load(dout_ptr + row * dout_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    # gradient through `weight * h`: computed in the output dtype, then cast to h's (= x's) dtype
    dn = (dout * w).to(dout_ptr.dtype.element_ty).to(x_ptr.dtype.element_ty).to(tl.float32)
    # gradient through n = x * rsqrt(mean(x^2) + eps)
    dot = tl.sum(dn * x, axis=0) / n_cols
    dx = rstd * dn - x * (rstd * rstd * rstd * dot)
    if ADD_DSUM:
        dsum = tl.load(dsum_ptr + row * dsum_row_stride + cols, mask=inside, other=0.0)
        dx = dx + dsum.to(tl.float32)
    tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    warps = 8 if block >= 2048 else 4 if block >= 512 else 2
    return block, warps


def _rows(t, n_cols):
    """A 2-D view (rows, n_cols) whose last dimension is contiguous."""
    t = t.reshape(-1, n_cols)
    return t if t.stride(-1) == 1 else t.contiguous()


def _norm_backward(dout2, dsum2, x2, weight, rstd, dx_dtype):
    """Input gradient of the norm (plus dsum2 when given), with x2 the norm's input."""
    n_cols = x2.shape[-1]
    dx = torch.empty(x2.shape, dtype=dx_dtype, device=x2.device)
    block, warps = _launch_config(n_cols)
    add = dsum2 is not None
    rmsnorm_bwd_kernel[(x2.shape[0],)](
        dout2, dsum2 if add else dout2, x2, weight, rstd, dx, dout2.stride(0),
        dsum2.stride(0) if add else 0, x2.stride(0), dx.stride(0), n_cols,
        ADD_DSUM=add, BLOCK=block, num_warps=warps)
    return dx


def _weight_grad(dout2, x2, rstd, out_dtype, weight):
    """Gradient of a trainable norm weight (not used under LoRA, where the norms are frozen)."""
    normed = (x2.float() * rstd[:, None]).to(x2.dtype).to(out_dtype)
    return (dout2 * normed).sum(0).to(weight.dtype)


class TritonRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        n_cols = x.shape[-1]
        x2 = _rows(x, n_cols)
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        out = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        block, warps = _launch_config(n_cols)
        rmsnorm_fwd_kernel[(x2.shape[0],)](
            x2, x2, x2, weight, out, rstd, x2.stride(0), 0, x2.stride(0), out.stride(0),
            n_cols, eps, ADD=False, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.x_shape, ctx.out_dtype = x.shape, out_dtype
        return out.view(*x.shape[:-1], n_cols)

    @staticmethod
    def backward(ctx, dout):
        x2, weight, rstd = ctx.saved_tensors
        dout2 = _rows(dout.to(ctx.out_dtype), x2.shape[-1])
        dx = dweight = None
        if ctx.needs_input_grad[0]:
            dx = _norm_backward(dout2, None, x2, weight, rstd, x2.dtype).view(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            dweight = _weight_grad(dout2, x2, rstd, ctx.out_dtype, weight)
        return dx, dweight, None


class TritonAddRMSNorm(torch.autograd.Function):
    """(residual, delta) -> (h = residual + delta, rmsnorm(h)) in one kernel each way."""

    @staticmethod
    def forward(ctx, residual, delta, weight, eps):
        n_cols = residual.shape[-1]
        r2, d2 = _rows(residual, n_cols), _rows(delta, n_cols)
        sum_dtype = torch.promote_types(residual.dtype, delta.dtype)
        out_dtype = torch.promote_types(weight.dtype, sum_dtype)
        h = torch.empty(r2.shape, dtype=sum_dtype, device=residual.device)
        out = torch.empty(r2.shape, dtype=out_dtype, device=residual.device)
        rstd = torch.empty(r2.shape[0], dtype=torch.float32, device=residual.device)
        block, warps = _launch_config(n_cols)
        rmsnorm_fwd_kernel[(r2.shape[0],)](
            r2, d2, h, weight, out, rstd, r2.stride(0), d2.stride(0), h.stride(0), out.stride(0),
            n_cols, eps, ADD=True, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(h, weight, rstd)
        ctx.shape, ctx.out_dtype = residual.shape, out_dtype
        ctx.dtypes = (residual.dtype, delta.dtype)
        return h.view(residual.shape), out.view(residual.shape)

    @staticmethod
    def backward(ctx, dh, dout):
        h, weight, rstd = ctx.saved_tensors
        n_cols = h.shape[-1]
        dout2 = _rows(dout.to(ctx.out_dtype), n_cols)
        dh2 = None if dh is None else _rows(dh.to(h.dtype), n_cols)
        dweight = None
        dsum = _norm_backward(dout2, dh2, h, weight, rstd, h.dtype).view(ctx.shape)
        if ctx.needs_input_grad[2]:
            dweight = _weight_grad(dout2, h, rstd, ctx.out_dtype, weight)
        # `residual + delta` passes the same gradient to both inputs (as PyTorch's add does)
        dres = dsum if dsum.dtype == ctx.dtypes[0] else dsum.to(ctx.dtypes[0])
        ddelta = dsum if dsum.dtype == ctx.dtypes[1] else dsum.to(ctx.dtypes[1])
        return dres, ddelta, dweight, None


def rmsnorm(x, weight, eps):
    return TritonRMSNorm.apply(x, weight, eps)


def add_rmsnorm(residual, delta, weight, eps):
    return TritonAddRMSNorm.apply(residual, delta, weight, eps)


def _fits(t):
    return t.is_cuda and t.shape[-1] <= 16384


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernels (a class patch, like the example)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "is_soup_triton", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if not _fits(hidden_states):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    forward.is_soup_triton = True
    forward.original = original
    cls.forward = forward


def patch_qwen2_decoder_add_norm():
    """Qwen2DecoderLayer.forward with `residual + attention` and post_attention_layernorm fused.
    Everything else is transformers' forward (v5.16.1) unchanged."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    if getattr(cls.forward, "is_soup_triton", False):
        return
    original = cls.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if not _fits(hidden_states):
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        norm = self.post_attention_layernorm
        residual, hidden_states = add_rmsnorm(residual, hidden_states, norm.weight,
                                              norm.variance_epsilon)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    forward.is_soup_triton = True
    forward.original = original
    cls.forward = forward
