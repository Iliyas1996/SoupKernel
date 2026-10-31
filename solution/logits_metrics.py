"""Training metrics straight from the fp16 logits (our own Triton kernel), so no fp32 copy of the
152k-wide logits is ever made just for logging.

Why (memory): after the loss, three readers still look at the logits of every step:
- accelerate's mixed-precision wrapper converts the model's fp16 outputs to fp32 on the way out
  (`convert_outputs_to_fp32`): +1.2 GB at 4 x 512 tokens;
- TRL logs the mean token entropy (`entropy_from_logits`) and accuracy (argmax);
- Soup's rewind log stores each row's mean token cross-entropy (`row_losses`).
Our diagnostics showed the step's memory peak right there.

What we do:
- keep the logits fp16 when they leave the model (every other output is still converted);
- compute TRL's per-token entropy and Soup's per-row loss with one Triton kernel that reads the
  fp16 logits and does all the math in fp32, the precision these metrics had before (they used to
  read the fp32 copy, which held exactly the same values);
- the accuracy is an argmax, which is the same on fp16 and on its exact fp32 copy.

Per token, in one pass over the vocabulary (online softmax, Milakov & Gimelshein 2018):
    m = running max,  s = sum exp(x - m),  t = sum exp(x - m) * x
    logsumexp = m + log s,   entropy = logsumexp - t / s,   loss = logsumexp - x[target]
(entropy = -sum p log p = logsumexp - sum p x). The kernel code is our own.
"""

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

VOCAB_BLOCK = 4096


@triton.jit
def token_entropy_loss_kernel(logits_ptr, target_ptr, entropy_ptr, loss_ptr, row_stride, n_cols,
                              ignore_index, WANT_ENTROPY: tl.constexpr, WANT_LOSS: tl.constexpr,
                              BLOCK: tl.constexpr):
    """One program: one token. Entropy of its softmax and/or its cross-entropy, in fp32."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * row_stride
    offs = tl.arange(0, BLOCK)
    m = tl.max(tl.full([BLOCK], float("-inf"), tl.float32), axis=0)
    s = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    t = tl.sum(tl.zeros([BLOCK], tl.float32), axis=0)
    for start in range(0, n_cols, BLOCK):
        cols = start + offs
        inside = cols < n_cols
        x = tl.load(row_ptr + cols, mask=inside, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        rescale = tl.exp(m - m_new)
        e = tl.exp(x - m_new)  # 0 outside the row
        s = s * rescale + tl.sum(e, axis=0)
        if WANT_ENTROPY:
            t = t * rescale + tl.sum(tl.where(inside, e * x, 0.0), axis=0)
        m = m_new
    lse = m + tl.log(s)
    if WANT_ENTROPY:
        tl.store(entropy_ptr + row, lse - t / s)
    if WANT_LOSS:
        target = tl.load(target_ptr + row)
        valid = target != ignore_index
        x_t = tl.load(row_ptr + tl.where(valid, target, 0)).to(tl.float32)
        tl.store(loss_ptr + row, tl.where(valid, lse - x_t, 0.0))


def _flat(logits):
    flat = logits.reshape(-1, logits.shape[-1])
    return flat if flat.stride(-1) == 1 else flat.contiguous()


def token_entropy(logits):
    """Per-token softmax entropy in fp32, shape logits.shape[:-1]."""
    flat = _flat(logits)
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    token_entropy_loss_kernel[(flat.shape[0],)](
        flat, out, out, out, flat.stride(0), flat.shape[1], -100,
        WANT_ENTROPY=True, WANT_LOSS=False, BLOCK=VOCAB_BLOCK, num_warps=8)
    return out.view(logits.shape[:-1])


def token_losses(logits, targets):
    """Per-token cross-entropy in fp32 (0 where target == -100), shape logits.shape[:-1]."""
    flat = _flat(logits)
    targets = targets.reshape(-1).to(device=logits.device, dtype=torch.int64).contiguous()
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    token_entropy_loss_kernel[(flat.shape[0],)](
        flat, targets, out, out, flat.stride(0), flat.shape[1], -100,
        WANT_ENTROPY=False, WANT_LOSS=True, BLOCK=VOCAB_BLOCK, num_warps=8)
    return out.view(logits.shape[:-1])


def _usable(logits):
    return (torch.is_tensor(logits) and logits.is_cuda and logits.dim() >= 2
            and logits.dtype in (torch.float16, torch.bfloat16, torch.float32))


def patch_trl_entropy():
    """TRL's SFTTrainer calls `entropy_from_logits` by name from its module."""
    from trl.trainer import sft_trainer

    original = sft_trainer.entropy_from_logits
    if getattr(original, "is_soup_triton", False):
        return

    def entropy_from_logits(logits, chunk_size=128):
        if not _usable(logits):
            return original(logits, chunk_size)
        return token_entropy(logits)

    entropy_from_logits.is_soup_triton = True
    entropy_from_logits.original = original
    sft_trainer.entropy_from_logits = entropy_from_logits


def patch_soup_row_losses():
    """Soup's rewind recorder calls `row_losses(logits, labels)` by name from its module.
    Same result as Soup 0.77.0's row_losses: per row, the mean over supervised tokens of the
    cross-entropy of logits[:, :-1] against labels[:, 1:], and the number of those tokens."""
    from soup_cli.trainer import rewind_hf

    original = rewind_hf.row_losses
    if getattr(original, "is_soup_triton", False):
        return

    def row_losses(logits, labels):
        if not _usable(logits) or logits.dim() != 3:
            return original(logits, labels)
        with torch.no_grad():
            targets = torch.full(labels.shape, -100, dtype=torch.int64, device=logits.device)
            targets[:, :-1] = labels[:, 1:]
            token_loss = token_losses(logits, targets)
            mask = (targets != -100).to(torch.float32)
            tokens = mask.sum(dim=1)
            totals = (token_loss * mask).sum(dim=1)
            means = (totals / tokens.clamp(min=1.0)).tolist()
            counts = tokens.to(torch.int64).tolist()
        return [float(v) for v in means], [int(c) for c in counts]

    row_losses.is_soup_triton = True
    row_losses.original = original
    rewind_hf.row_losses = row_losses


def patch_keep_fp16_logits():
    """accelerate (1.14) wraps the model's forward in ConvertOutputsToFp32 under fp16 mixed
    precision. Keep the big `logits` tensor as it is; convert every other output as before."""
    from accelerate.utils import operations

    cls = operations.ConvertOutputsToFp32
    if getattr(cls.__call__, "is_soup_triton", False):
        return
    original_call = cls.__call__

    def __call__(self, *args, **kwargs):
        outputs = self.model_forward(*args, **kwargs)
        logits = outputs.get("logits") if isinstance(outputs, Mapping) else None
        if not (_usable(logits) and logits.dim() == 3):
            return operations.convert_to_fp32(outputs)
        converted = operations.convert_to_fp32({k: v for k, v in outputs.items() if k != "logits"})
        for key, value in converted.items():  # update in place: same type, same key order
            outputs[key] = value
        return outputs

    __call__.is_soup_triton = True
    __call__.original = original_call
    cls.__call__ = __call__


def patch_logits_metrics():
    """All three together: they only make sense together (fp16 logits + fp32-exact metrics)."""
    try:
        patch_trl_entropy()
        patch_soup_row_losses()
    except Exception:  # noqa: BLE001 - a different TRL/Soup: keep the fp32 copy, change nothing
        return
    patch_keep_fp16_logits()
