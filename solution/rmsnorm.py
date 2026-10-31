"""Our own fused RMSNorm (Triton), forward and backward, for Qwen2RMSNorm.

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

The math is the standard RMSNorm derivation (Zhang & Sennrich 2019, "Root Mean Square Layer
Normalization"); fusing it into one kernel per direction is a common idea (for example in the
Triton layer-norm tutorial); the kernel code here is our own.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, out_ptr, rstd_ptr, x_row_stride, out_row_stride, n_cols, eps,
                       BLOCK: tl.constexpr):
    """One program normalizes one row (one token)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(rstd_ptr + row, rstd)
    # `.to(input_dtype)` in transformers rounds the normalized value to the input dtype
    normed = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * out_row_stride + cols, (w * normed).to(out_ptr.dtype.element_ty),
             mask=inside)


@triton.jit
def rmsnorm_bwd_kernel(dout_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dout_row_stride, x_row_stride,
                       dx_row_stride, n_cols, BLOCK: tl.constexpr):
    """One program computes the input gradient of one row."""
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
    tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    warps = 8 if block >= 2048 else 4 if block >= 512 else 2
    return block, warps


def _rows(t, n_cols):
    """A 2-D view (rows, n_cols) whose last dimension is contiguous."""
    t = t.reshape(-1, n_cols)
    return t if t.stride(-1) == 1 else t.contiguous()


class TritonRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        n_cols = x.shape[-1]
        x2 = _rows(x, n_cols)
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        out = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        block, warps = _launch_config(n_cols)
        rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, out, rstd, x2.stride(0), out.stride(0),
                                           n_cols, eps, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.x_shape, ctx.out_dtype = x.shape, out_dtype
        return out.view(*x.shape[:-1], n_cols)

    @staticmethod
    def backward(ctx, dout):
        x2, weight, rstd = ctx.saved_tensors
        n_cols = x2.shape[-1]
        dout2 = _rows(dout.to(ctx.out_dtype), n_cols)
        dx = dweight = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x2.shape, dtype=x2.dtype, device=x2.device)
            block, warps = _launch_config(n_cols)
            rmsnorm_bwd_kernel[(x2.shape[0],)](dout2, x2, weight, rstd, dx, dout2.stride(0),
                                               x2.stride(0), dx.stride(0), n_cols,
                                               BLOCK=block, num_warps=warps)
            dx = dx.view(ctx.x_shape)
        if ctx.needs_input_grad[1]:  # not used under LoRA (the norms are frozen); plain PyTorch
            normed = (x2.float() * rstd[:, None]).to(x2.dtype).to(ctx.out_dtype)
            dweight = (dout2 * normed).sum(0).to(weight.dtype)
        return dx, dweight, None


def rmsnorm(x, weight, eps):
    return TritonRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernels (a class patch, like the example)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "is_soup_triton", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda or hidden_states.shape[-1] > 16384:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    forward.is_soup_triton = True
    forward.original = original
    cls.forward = forward
