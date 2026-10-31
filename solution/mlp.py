"""Qwen2 MLP that keeps its wide intermediate tensors in the 4-bit layers' compute dtype (fp16).

Why (memory): in this task the model's hidden states are fp32 while bitsandbytes computes in fp16.
Each bitsandbytes Linear4bit casts its input to fp16, multiplies, and casts the result back to
fp32. So gate_proj and up_proj hand fp32 tensors of shape (tokens, 8960) to the SwiGLU, which
keeps both for its backward pass: about 147 MB per layer, 4.1 GB over 28 layers at 2048 tokens.
Those fp32 values are exact copies of fp16 numbers, so nothing is lost by keeping them in fp16.

What changes: gate_proj and up_proj results stay fp16, the SwiGLU reads fp16 and writes fp16, and
only down_proj's result is cast back to the input dtype, exactly as Linear4bit would do it.
Why it is bit-identical: the SwiGLU kernel computes in fp32 either way; writing its result as fp16
here is the same single rounding that down_proj's `.to(fp16)` applied before. In the backward pass
the same holds for the gradients (every gradient that reached a 4-bit layer was rounded to fp16).
The input x is cast to fp16 separately for gate_proj and up_proj, as before, so the two input
gradients are still added in fp32.

The SwiGLU itself is the organizers' example kernel (solution/swiglu.py, it does not count as our
own); this file only changes the dtypes around it and does not launch kernels of its own.
"""

import torch

from .swiglu import patch_qwen2_mlp, swiglu


def _linear4bit_compute_dtype(module, x):
    """bitsandbytes 0.50.2 Linear4bit.forward without its final `.to(input dtype)`."""
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

    fix_4bit_weight_quant_state_from_module(module)
    if not module.compute_type_is_set:
        module.set_compute_type(x)
        module.compute_type_is_set = True
    if module.compute_dtype is not None:
        x = x.to(module.compute_dtype)
    bias = module.bias
    if bias is not None:
        if bias.dtype != x.dtype:
            bias.data = bias.data.to(x.dtype)
        bias = bias.to(module.compute_dtype)
    return bnb.matmul_4bit(x, module.weight, bias=bias, quant_state=module.weight.quant_state)


def _plain_linear4bit(module):
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return type(module) is bnb.nn.Linear4bit and \
        getattr(module.weight, "quant_state", None) is not None


def patch_qwen2_mlp_low_precision():
    from transformers.models.qwen2 import modeling_qwen2

    patch_qwen2_mlp()  # the example's patch; we fall back to it whenever our path does not apply
    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "is_soup_lowp", False):
        return
    fallback = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not x.is_cuda or not torch.is_grad_enabled()
                or not all(_plain_linear4bit(m) for m in (self.gate_proj, self.up_proj,
                                                           self.down_proj))):
            return fallback(self, x)
        gate = _linear4bit_compute_dtype(self.gate_proj, x)
        up = _linear4bit_compute_dtype(self.up_proj, x)
        if gate.dtype != up.dtype:
            return fallback(self, x)
        hidden = swiglu(gate, up)
        return _linear4bit_compute_dtype(self.down_proj, hidden).to(x.dtype)

    forward.is_soup_lowp = True
    forward.fallback = fallback
    cls.forward = forward
