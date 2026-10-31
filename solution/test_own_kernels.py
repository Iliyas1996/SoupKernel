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
