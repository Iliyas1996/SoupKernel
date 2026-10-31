"""Our own fused causal-LM cross-entropy (Triton) over the 151,936-token vocabulary.

Why (memory): transformers' ForCausalLMLoss first makes an fp32 copy of the fp16 logits
(`logits.float()`, 1.2 GB at 4 x 512 tokens), then log_softmax keeps another fp32 tensor of the
same size for the backward pass. Our forward kernel reads the fp16 logits directly and keeps only
two fp32 numbers per token (log-sum-exp and the token loss); the backward kernel recomputes the
softmax from the same fp16 logits and writes the fp16 gradient directly.

What it computes, exactly as transformers 5.16.1 ForCausalLMLoss (+ fixed_cross_entropy):
    targets  = labels shifted left by one, the last position = ignore_index
    loss_t   = logsumexp(logits_t) - logits_t[target_t]          (fp32; 0 where ignored)
    loss     = sum_t loss_t / num_items_in_batch                  (or / #targets if not given)
    dlogits  = (softmax(logits_t) - onehot(target_t)) * dloss / num_items_in_batch,
               rounded once to the logits' dtype (as the `.float()` cast's backward does).

The log-sum-exp is computed in one pass over the vocabulary with a running maximum ("online
softmax", Milakov & Gimelshein 2018, "Online normalizer calculation for softmax"); computing the
gradient inside a loss kernel is the idea behind fused-loss kernels such as Liger's (Hsu et al.
2024) and Apple's Cut Cross-Entropy (Wijmans et al. 2024). The kernel code here is our own.
"""

import torch
import triton
import triton.language as tl

VOCAB_BLOCK = 4096


@triton.jit
def ce_lse_loss_kernel(logits_ptr, target_ptr, lse_ptr, loss_ptr, row_stride, n_cols,
                       ignore_index, BLOCK: tl.constexpr):
    """One program: one token. Log-sum-exp over the vocabulary in BLOCK-wide chunks, then the
    token's loss."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * row_stride
    offs = tl.arange(0, BLOCK)
    running_max = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    running_sum = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        x = tl.load(row_ptr + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(running_max, tl.max(x, axis=0))
        running_sum = running_sum * tl.exp(running_max - new_max) + \
            tl.sum(tl.exp(x - new_max), axis=0)
        running_max = new_max
    lse = running_max + tl.log(running_sum)
    tl.store(lse_ptr + row, lse)
    target = tl.load(target_ptr + row)
    valid = target != ignore_index
    safe_target = tl.where(valid, target, 0)
    target_logit = tl.load(row_ptr + safe_target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))


@triton.jit
def ce_grad_kernel(logits_ptr, target_ptr, lse_ptr, scale_ptr, grad_ptr, row_stride,
                   grad_row_stride, n_cols, ignore_index, BLOCK: tl.constexpr):
    """One program: one token. grad = (softmax - onehot(target)) * scale, 0 for ignored tokens."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * row_stride
    out_ptr = grad_ptr + row * grad_row_stride
    target = tl.load(target_ptr + row)
    lse = tl.load(lse_ptr + row)
    scale = tl.where(target != ignore_index, tl.load(scale_ptr), 0.0)
    offs = tl.arange(0, BLOCK)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        inside = cols < n_cols
        x = tl.load(row_ptr + cols, mask=inside, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        g = tl.where(cols == target, p - 1.0, p) * scale
        tl.store(out_ptr + cols, g.to(grad_ptr.dtype.element_ty), mask=inside)


class TritonCausalLMLoss(torch.autograd.Function):
    """logits (N, V) any float dtype with a contiguous last dim; targets (N,) int64;
    denom: 0-dim fp32 tensor (num_items_in_batch or the number of targets)."""

    @staticmethod
    def forward(ctx, logits, targets, denom):
        n_rows, n_cols = logits.shape
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        token_loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        ce_lse_loss_kernel[(n_rows,)](logits, targets, lse, token_loss, logits.stride(0), n_cols,
                                      -100, BLOCK=VOCAB_BLOCK, num_warps=8)
        ctx.save_for_backward(logits, targets, lse, denom)
        return token_loss.sum() / denom

    @staticmethod
    def backward(ctx, grad_loss):
        logits, targets, lse, denom = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        scale = (grad_loss.float() / denom).reshape(1)
        grad = torch.empty(logits.shape, dtype=logits.dtype, device=logits.device)
        ce_grad_kernel[(n_rows,)](logits, targets, lse, scale, grad, logits.stride(0),
                                  grad.stride(0), n_cols, -100, BLOCK=VOCAB_BLOCK, num_warps=8)
        return grad, None, None


def causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                   shift_labels=None):
    if shift_labels is None:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:]
    logits2 = logits.reshape(-1, vocab_size)
    if logits2.stride(-1) != 1:
        logits2 = logits2.contiguous()
    targets = shift_labels.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()
    if num_items_in_batch is not None:
        denom = torch.as_tensor(num_items_in_batch, device=logits.device).to(torch.float32)
    else:  # reduction "mean" with ignore_index: divide by the number of real targets
        denom = (targets != ignore_index).sum().to(torch.float32)
    return TritonCausalLMLoss.apply(logits2, targets, denom.reshape(()))


def patch_causal_lm_loss():
    """Replace the default causal-LM loss transformers looks up in LOSS_MAPPING at every call."""
    from transformers.loss import loss_utils

    original = loss_utils.LOSS_MAPPING["ForCausalLM"]
    if getattr(original, "is_soup_triton", False):
        return

    def for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                           shift_labels=None, **kwargs):
        if (not logits.is_cuda or ignore_index != -100 or logits.shape[-1] != vocab_size
                or labels is None):
            return original(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
        return causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                              ignore_index=ignore_index, shift_labels=shift_labels)

    for_causal_lm_loss.is_soup_triton = True
    for_causal_lm_loss.original = original
    loss_utils.LOSS_MAPPING["ForCausalLM"] = for_causal_lm_loss
