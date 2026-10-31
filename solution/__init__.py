"""Our solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` swaps a few library functions for versions
that run our own Triton kernels:

- solution/nf4_dequant.py: NF4 4-bit weight unpacking (bitsandbytes' dequantize) in Triton;
- solution/rmsnorm.py:     fused RMSNorm forward/backward for every Qwen2RMSNorm, and a fused
                           "residual add + RMSNorm" after each attention block;
- solution/rope.py:        fused rotary position embedding for the queries and keys;
- solution/swiglu.py:      the organizers' example (fused SwiGLU); kept because it saves memory,
                           it does not count as our own kernel.
"""

from .nf4_dequant import patch_bnb_dequant
from .rmsnorm import patch_qwen2_decoder_add_norm, patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope
from .swiglu import patch_qwen2_mlp


def apply():
    patch_bnb_dequant()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_add_norm()
    patch_qwen2_rope()
    patch_qwen2_mlp()  # the example: works out of the box, does not count as our own kernel

    from . import diagnostics

    if diagnostics.enabled():  # only for our own analysis runs (SOUP_KERNEL_DIAG=1), never measured
        diagnostics.enable()
