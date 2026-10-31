"""Correctness of our own kernels against the code they replace, forward and backward.

Run on a GPU: `python -m pytest solution -q` (the Kaggle notebook runs it before training).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


# ---------------------------------------------------------------- NF4 dequantization

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536), (37, 64)])
def test_nf4_dequant_is_bit_exact(dtype, nested, shape):
    import bitsandbytes.functional as F

    from solution.nf4_dequant import patch_bnb_dequant, unpatch_bnb_dequant

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=dtype)
    packed, state = F.quantize_4bit(weight, blocksize=64, compress_statistics=nested,
                                    quant_type="nf4")
    unpatch_bnb_dequant()
    reference = F.dequantize_4bit(packed, state)       # bitsandbytes' own CUDA kernel
    patch_bnb_dequant()
    try:
        ours = F.dequantize_4bit(packed, state)        # the same call, now through our kernel
    finally:
        unpatch_bnb_dequant()
    assert ours.dtype == reference.dtype and ours.shape == reference.shape
    assert torch.equal(ours, reference)


def test_nf4_matmul_4bit_forward_and_backward():
    """The full bitsandbytes 4-bit linear (forward + input gradient) gives identical results."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as F

    from solution.nf4_dequant import patch_bnb_dequant, unpatch_bnb_dequant

    torch.manual_seed(0)
    weight = torch.randn(1536, 1536, device="cuda", dtype=torch.float16)
    packed, state = F.quantize_4bit(weight, blocksize=64, quant_type="nf4")
    x = torch.randn(4, 300, 1536, device="cuda", dtype=torch.float16)
    grad = torch.randn(4, 300, 1536, device="cuda", dtype=torch.float16)

    def run():
        a = x.clone().requires_grad_(True)
        out = bnb.matmul_4bit(a, packed.t(), quant_state=state)
        out.backward(grad)
        return out.detach(), a.grad

    unpatch_bnb_dequant()
    ref_out, ref_grad = run()
    patch_bnb_dequant()
    try:
        out, grad_a = run()
    finally:
        unpatch_bnb_dequant()
    assert torch.equal(out, ref_out)
    assert torch.equal(grad_a, ref_grad)


# ---------------------------------------------------------------- RMSNorm

def reference_rmsnorm(x, weight, eps):
    """transformers' Qwen2RMSNorm.forward, line for line."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


TOL = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("x_dtype,w_dtype", [(torch.float32, torch.float32),
                                             (torch.float16, torch.float32),
                                             (torch.float16, torch.float16),
                                             (torch.float32, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 300, 1536), (3, 77, 1536), (2, 5, 1000)])
def test_rmsnorm_matches_qwen2(x_dtype, w_dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=x_dtype)
    weight = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).to(w_dtype)
    grad = torch.randn(shape, device="cuda").to(torch.promote_types(x_dtype, w_dtype))

    xr = x.clone().requires_grad_(True)
    ref = reference_rmsnorm(xr, weight, eps)
    ref.backward(grad)

    xo = x.clone().requires_grad_(True)
    out = rmsnorm(xo, weight, eps)
    out.backward(grad)

    tol = max(TOL[x_dtype], TOL[w_dtype])
    assert out.dtype == ref.dtype and out.shape == ref.shape
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    assert xo.grad.dtype == xr.grad.dtype
    torch.testing.assert_close(xo.grad.float(), xr.grad.float(), rtol=tol, atol=tol)


def test_rmsnorm_weight_gradient():
    """The norm weights are frozen under LoRA, but a trainable weight still gets its gradient."""
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(2, 50, 1536, device="cuda")
    grad = torch.randn(2, 50, 1536, device="cuda")
    w_ref = (1 + 0.1 * torch.randn(1536, device="cuda")).requires_grad_(True)
    w_own = w_ref.detach().clone().requires_grad_(True)
    reference_rmsnorm(x, w_ref, 1e-6).backward(grad)
    rmsnorm(x, w_own, 1e-6).backward(grad)
    torch.testing.assert_close(w_own.grad, w_ref.grad, rtol=1e-4, atol=1e-3)


def test_patch_routes_qwen2_rmsnorm_through_triton():
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rmsnorm import patch_qwen2_rmsnorm

    norm = modeling_qwen2.Qwen2RMSNorm(1536, eps=1e-6).cuda()
    x = torch.randn(2, 10, 1536, device="cuda")
    expected = norm(x)
    original = modeling_qwen2.Qwen2RMSNorm.forward
    patch_qwen2_rmsnorm()
    try:
        assert getattr(modeling_qwen2.Qwen2RMSNorm.forward, "is_soup_triton", False)
        torch.testing.assert_close(norm(x), expected, rtol=1e-4, atol=1e-4)
    finally:
        modeling_qwen2.Qwen2RMSNorm.forward = original


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_rmsnorm_matches_add_then_norm(dtype):
    from solution.rmsnorm import add_rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    shape = (4, 300, 1536)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    delta = torch.randn(shape, device="cuda", dtype=dtype)
    weight = 1 + 0.1 * torch.randn(1536, device="cuda")
    g_h = torch.randn(shape, device="cuda").to(dtype)
    g_out = torch.randn(shape, device="cuda")

    r1, d1 = residual.clone().requires_grad_(True), delta.clone().requires_grad_(True)
    h_ref = r1 + d1
    out_ref = reference_rmsnorm(h_ref, weight, eps)
    (h_ref * g_h).sum().backward(retain_graph=True)
    out_ref.backward(g_out)

    r2, d2 = residual.clone().requires_grad_(True), delta.clone().requires_grad_(True)
    h, out = add_rmsnorm(r2, d2, weight, eps)
    (h * g_h).sum().backward(retain_graph=True)
    out.backward(g_out)

    tol = TOL[dtype]
    assert torch.equal(h, h_ref)  # the residual itself is bit-exact
    torch.testing.assert_close(out, out_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(r2.grad.float(), r1.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(d2.grad.float(), d1.grad.float(), rtol=tol, atol=tol)


# ---------------------------------------------------------------- RoPE

def reference_rope(q, cos, sin):
    """transformers' apply_rotary_pos_emb for one tensor."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    half = q.shape[-1] // 2
    rotated = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    return (q * cos) + (rotated * sin)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("heads,tokens,batch_cos", [(12, 300, 4), (2, 300, 1), (12, 7, 1)])
def test_rope_matches_transformers(dtype, heads, tokens, batch_cos):
    from solution.rope import TritonRoPE

    torch.manual_seed(0)
    batch, dim = 4, 128
    # q as transformers makes it: (batch, tokens, heads, dim) viewed as (batch, heads, tokens, dim)
    q = torch.randn(batch, tokens, heads, dim, device="cuda", dtype=dtype).transpose(1, 2)
    angles = torch.rand(batch_cos, tokens, dim // 2, device="cuda") * 50
    emb = torch.cat((angles, angles), dim=-1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    grad = torch.randn(batch, heads, tokens, dim, device="cuda").to(dtype)

    q1 = q.detach().clone().requires_grad_(True)
    ref = reference_rope(q1, cos, sin)
    ref.backward(grad)

    q2 = q.detach().clone().requires_grad_(True)
    cos_b = cos.expand(batch, -1, -1) if batch_cos == 1 else cos
    sin_b = sin.expand(batch, -1, -1) if batch_cos == 1 else sin
    out = TritonRoPE.apply(q2, cos_b, sin_b)
    out.backward(grad)

    # PyTorch rounds fp16 after each of the three ops; the kernel rounds once at the end
    tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(q2.grad, q1.grad, rtol=tol, atol=tol)


# ---------------------------------------------------------------- a whole Qwen2 decoder layer

def test_patched_qwen2_decoder_layer_matches_original():
    """RMSNorm, add+RMSNorm and RoPE patches together, on a small random Qwen2 layer."""
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rmsnorm import patch_qwen2_decoder_add_norm, patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope

    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=256, intermediate_size=512, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1, vocab_size=100)
    config._attn_implementation = "sdpa"
    layer = modeling_qwen2.Qwen2DecoderLayer(config, layer_idx=0).cuda()
    with torch.no_grad():
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            norm.weight.add_(0.1 * torch.randn_like(norm.weight))
    rotary = modeling_qwen2.Qwen2RotaryEmbedding(config=config).cuda()
    x = torch.randn(2, 33, 256, device="cuda")
    position_ids = torch.arange(33, device="cuda")[None]
    grad = torch.randn(2, 33, 256, device="cuda")

    def run():
        layer.zero_grad()
        inp = x.clone().requires_grad_(True)
        position_embeddings = rotary(inp, position_ids)
        out = layer(inp, position_ids=position_ids, position_embeddings=position_embeddings)
        out.backward(grad)
        return out.detach(), inp.grad, layer.self_attn.q_proj.weight.grad.clone()

    saved = (modeling_qwen2.Qwen2RMSNorm.forward, modeling_qwen2.Qwen2DecoderLayer.forward,
             modeling_qwen2.apply_rotary_pos_emb)
    ref = run()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_add_norm()
    patch_qwen2_rope()
    try:
        assert getattr(modeling_qwen2.Qwen2DecoderLayer.forward, "is_soup_triton", False)
        ours = run()
    finally:
        (modeling_qwen2.Qwen2RMSNorm.forward, modeling_qwen2.Qwen2DecoderLayer.forward,
         modeling_qwen2.apply_rotary_pos_emb) = saved
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
