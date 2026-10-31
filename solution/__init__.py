"""Our solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` swaps a few library functions for versions
that run our own Triton kernels:

- solution/nf4_dequant.py: NF4 4-bit weight unpacking (bitsandbytes' dequantize) in Triton;
- solution/rmsnorm.py:     fused RMSNorm forward and backward for Qwen2RMSNorm;
- solution/swiglu.py:      the organizers' example (fused SwiGLU); kept because it saves memory,
                           it does not count as our own kernel.
"""

from .nf4_dequant import patch_bnb_dequant
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp


def apply():
    patch_bnb_dequant()
    patch_qwen2_rmsnorm()
    patch_qwen2_mlp()  # the example: works out of the box, does not count as our own kernel
