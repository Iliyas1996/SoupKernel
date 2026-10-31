"""Our own fused rotary position embedding (RoPE) for Qwen2 attention, forward and backward.

Why: transformers applies RoPE to the queries and keys with plain PyTorch:
    q_embed = q * cos + rotate_half(q) * sin,   rotate_half(q) = cat(-q2, q1)
That is a negation, a concatenation, two multiplies and an add for q and again for k, on strided
(transposed) tensors, and the same again in the backward pass: in our profile these show up as the
non-vectorized `mul`, `add` and `copy` kernels. Here it is one kernel per tensor and direction.

With the halves x1 = x[..., :D/2], x2 = x[..., D/2:] (same for cos c1, c2 and sin s1, s2):
    forward:   o1 = x1*c1 - x2*s1             o2 = x2*c2 + x1*s2
    backward:  dx1 = g1*c1 + g2*s2            dx2 = g2*c2 - g1*s1
RoPE itself is from Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position
Embedding"; the kernel code here is our own.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_rotate_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                       x_sb, x_sh, x_st, x_sd, o_sb, o_sh, o_st, o_sd, c_sb, c_st, c_sd,
                       BACKWARD: tl.constexpr, HALF: tl.constexpr):
    """One program: one (token, head, batch) row of D = 2 * HALF values.
    Forward rotates by +angle; BACKWARD rotates the gradient by -angle (the transpose)."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    d = tl.arange(0, HALF)
    x_row = x_ptr + b * x_sb + h * x_sh + t * x_st
    c_row = b * c_sb + t * c_st
    x1 = tl.load(x_row + d * x_sd).to(tl.float32)
    x2 = tl.load(x_row + (d + HALF) * x_sd).to(tl.float32)
    c1 = tl.load(cos_ptr + c_row + d * c_sd).to(tl.float32)
    c2 = tl.load(cos_ptr + c_row + (d + HALF) * c_sd).to(tl.float32)
    s1 = tl.load(sin_ptr + c_row + d * c_sd).to(tl.float32)
    s2 = tl.load(sin_ptr + c_row + (d + HALF) * c_sd).to(tl.float32)
    if BACKWARD:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    o_row = out_ptr + b * o_sb + h * o_sh + t * o_st
    tl.store(o_row + d * o_sd, y1.to(out_ptr.dtype.element_ty))
    tl.store(o_row + (d + HALF) * o_sd, y2.to(out_ptr.dtype.element_ty))


def _rotate(x, cos, sin, out, backward):
    batch, heads, tokens, dim = x.shape
    rope_rotate_kernel[(tokens, heads, batch)](
        x, cos, sin, out, *x.stride(), *out.stride(), *cos.stride(),
        BACKWARD=backward, HALF=dim // 2, num_warps=1)
    return out


class TritonRoPE(torch.autograd.Function):
    """x: (batch, heads, tokens, dim) in any layout; cos, sin: (batch, tokens, dim)."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, cos.dtype))
        _rotate(x, cos, sin, out, backward=False)
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        dx = torch.empty_like(grad, dtype=ctx.x_dtype)
        _rotate(grad, cos, sin, dx, backward=True)
        return dx, None, None


def _supported(x, cos):
    dim = x.shape[-1]
    return (x.is_cuda and x.dim() == 4 and cos.dim() == 3 and dim % 2 == 0
            and dim // 2 == triton.next_power_of_2(dim // 2)
            and cos.shape[0] in (1, x.shape[0]) and cos.shape[1] == x.shape[2]
            and cos.shape[2] == dim)


def patch_qwen2_rope():
    """Replace transformers' apply_rotary_pos_emb for Qwen2 (looked up by name at every call)."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "is_soup_triton", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if (unsqueeze_dim != 1 or cos.requires_grad or sin.requires_grad
                or not (_supported(q, cos) and _supported(k, cos)) or sin.shape != cos.shape):
            return original(q, k, cos, sin, unsqueeze_dim)
        if cos.shape[0] != q.shape[0]:  # one set of positions shared by the whole batch
            cos, sin = cos.expand(q.shape[0], -1, -1), sin.expand(q.shape[0], -1, -1)
        return TritonRoPE.apply(q, cos, sin), TritonRoPE.apply(k, cos, sin)

    apply_rotary_pos_emb.is_soup_triton = True
    apply_rotary_pos_emb.original = original
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
