"""Our solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` swaps a few library functions for versions
that run our own Triton kernels:

- solution/nf4_dequant.py: NF4 4-bit weight unpacking (bitsandbytes' dequantize) in Triton;
- solution/rmsnorm.py:     fused RMSNorm forward/backward for every Qwen2RMSNorm, and a fused
                           "residual add + RMSNorm" after each attention block;
- solution/rope.py:        fused rotary position embedding for the queries and keys;
- solution/cross_entropy.py: fused causal-LM cross-entropy over the 152k vocabulary, straight
                           from the fp16 logits (no fp32 copies);
- solution/logits_metrics.py: TRL's entropy and Soup's per-row loss computed in fp32 by our kernel
                           straight from the fp16 logits, so the logits are not copied to fp32
                           just for logging;
- solution/mlp.py:         keeps the MLP's wide intermediates in fp16 (bit-identical, half the
                           memory) around the organizers' example SwiGLU (solution/swiglu.py),
                           which does not count as our own kernel.
"""

from .cross_entropy import patch_causal_lm_loss
from .logits_metrics import patch_logits_metrics
from .mlp import patch_qwen2_mlp_low_precision
from .nf4_dequant import patch_bnb_dequant
from .rmsnorm import patch_qwen2_decoder_add_norm, patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope


def apply():
    patch_bnb_dequant()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_add_norm()
    patch_qwen2_rope()
    patch_causal_lm_loss()
    patch_logits_metrics()
    patch_qwen2_mlp_low_precision()  # wraps the example's SwiGLU patch

    from . import diagnostics

    if diagnostics.enabled():  # only for our own analysis runs (SOUP_KERNEL_DIAG=1), never measured
        diagnostics.enable()
