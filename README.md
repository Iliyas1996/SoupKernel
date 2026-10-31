# Soup Kernel Challenge: our solution

Seven Triton kernels of our own, plugged into `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct,
NF4 4-bit, LoRA r=16, Kaggle T4) through `solution.apply()`. Same task, same adapter quality,
**1.94x less peak GPU memory** and **1.13x faster**.

| Kaggle T4, one run each (`harness/run_task.py`) | Base Soup | Ours | Ratio |
|---|---|---|---|
| Full `soup train` time (start → saved adapter) | 76.7 s | 67.6 s | **speedup 1.135x** |
| Peak GPU memory (nvidia-smi, every 0.1 s) | 14,345 MiB | 7,381 MiB | **memory 1.944x** |
| Eval loss on the public rows 40000–40255 | 2.04967 | 2.04966 | −0.000% |
| Our own kernels, share of training time (check 4) | — | 10.0% | ≥ 5% required |

All 10 checks of `harness/check_rules.py` pass, and the 53 tests in `solution/` pass on the T4.

- **Kaggle notebook (public):** https://www.kaggle.com/code/ilyasasilgaly/notebook639bc5ef9a
  (the saved version with results, and the exact final commit, are linked in
  [SUBMISSION.md](SUBMISSION.md))
- **Repository:** https://github.com/Iliyas1996/SoupKernel

## What we did and why

We started from a profile of the base run (`run_task.py --profile`) and from a memory timeline
of one training step (our opt-in `solution/diagnostics.py`, see "How to run"). Two findings
drove everything:

1. **The model's hidden states are fp32, while bitsandbytes computes in fp16.** Every 4-bit layer
   casts its input to fp16 and its output back to fp32, so many activations kept for the backward
   pass are fp32 copies of fp16 values, and about 18% of the GPU time is plain `copy` kernels.
2. **The 151,936-token vocabulary.** At 4 × 512 tokens the fp16 logits are 0.6 GB. The loss made
   an fp32 copy of them (1.2 GB) and kept an fp32 `log_softmax` (1.2 GB); accelerate converted
   the logits to fp32 again on the way out of the model (1.2 GB), and TRL's accuracy metric copied
   those (1.2 GB).

Memory of one training step (MiB allocated by PyTorch, step 4, batch 4 × 512 tokens, from
`solution/diagnostics.py`):

| Point in the step | Starter kit (example SwiGLU only) | After the MLP + loss changes |
|---|---|---|
| Weights, step start | 1,578 | 1,578 |
| After the 28 decoder layers | 7,419 | 5,459 |
| After lm_head + loss | 9,633 | 6,486 |
| After leaving the model (accelerate's fp32 conversion) | 10,226 | 7,673 |
| Step peak (during TRL's entropy and accuracy) | 11,413 | 8,857 |
| **Peak seen by nvidia-smi (scored)** | **12,381** | **9,161** (final version: **7,381**) |

### Our kernels

| `@triton.jit` kernel | File | What it replaces | Effect |
|---|---|---|---|
| `nf4_unpack_scale_kernel` | `solution/nf4_dequant.py` | bitsandbytes' NF4 dequantization (`kDequantizeBlockwise`), 389 launches per step, forward and backward | bit-identical weights; about 4% of the step time |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm (about 10 small kernels each way, 57 norms per forward) and the residual add before each `post_attention_layernorm` (the forward add and the backward gradient add) | one launch each way |
| `rope_rotate_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (neg, cat, 2 mul, add on strided q and k, and their backward) | one launch per tensor and direction |
| `ce_lse_loss_kernel`, `ce_grad_kernel` | `solution/cross_entropy.py` | transformers' `ForCausalLMLoss` (an fp32 copy of the logits and a saved fp32 log_softmax) | loss and its fp16 gradient straight from the fp16 logits |
| `token_entropy_loss_kernel` | `solution/logits_metrics.py` | TRL's `entropy_from_logits` and Soup's rewind `row_losses`, which read fp32 logits | the same fp32 math from the fp16 logits, so the logits stay fp16: −1.8 GB peak |

Plus one change with no kernel of its own, `solution/mlp.py`: the gate_proj and up_proj results
stay in the 4-bit layers' fp16 compute dtype around the organizers' example SwiGLU (which does not
count as ours and is kept running). This halves the MLP activations kept for the backward pass
(−2.0 GB after the decoder layers) and is bit-identical: the fp32 values it replaces were exact
copies of fp16 numbers, and the SwiGLU computes in fp32 either way.

### How the kernels work

- **NF4 dequantization.** One program writes 2,048 consecutive weights. Each byte holds two 4-bit
  codes (element 2i is the high nibble); a code indexes the 16-value NormalFloat table, and every
  64 weights share one fp32 scale (`absmax`): `w = TABLE[code] * absmax[i // 64]`, the same fp32
  multiply and rounding as bitsandbytes, so the result is bit-identical (tested with `torch.equal`,
  with and without double quantization, fp16 and fp32). We replace bitsandbytes' internal
  `_dequantize_4bit_impl`, which both the forward pass (`MatMul4Bit`) and the backward pass
  (`dequantize_4bit`) call; anything else (FP4, other dtypes) goes to the original.
- **RMSNorm (+ residual add).** One program per token row (all 1,536 values in one block): load
  the row, optionally add the residual and write the sum, compute `rstd = rsqrt(mean(x²) + eps)`,
  write `weight * (x·rstd → input dtype)`, and keep only `rstd` (one fp32 per row). Backward:
  `dx = rstd·dn − x·rstd³·mean(dn·x)` with `dn = dout·weight`, plus the gradient that reached the
  residual directly. The dtype roundings follow transformers' Qwen2RMSNorm line by line. The
  patched `Qwen2DecoderLayer.forward` is transformers 5.16.1's forward with the
  `residual + attention` add and `post_attention_layernorm` fused.
- **RoPE.** One program per (token, head, batch) row: with halves x1, x2 the forward is
  `(x1·c1 − x2·s1, x2·c2 + x1·s2)` and the backward is its transpose
  `(g1·c1 + g2·s2, g2·c2 − g1·s1)`. q and k are read in their transposed layout through strides,
  so no copy is made.
- **Cross-entropy.** Forward: one program per token streams its 151,936 logits in blocks of 4,096
  with a running maximum (online softmax) to get the log-sum-exp, and stores `lse − logit[target]`
  (0 for ignored labels); the loss is `sum / num_items_in_batch`, exactly transformers'
  `ForCausalLMLoss` + `fixed_cross_entropy`. Backward: one pass recomputes `softmax − onehot`,
  scales it by `dloss / num_items_in_batch` and writes the fp16 gradient with a single rounding, as
  the backward of `logits.float()` did. Only the fp16 logits (the model's own output tensor) and
  two fp32 numbers per token are kept.
- **Metrics from fp16 logits.** One pass per token keeps the running max `m`, `s = Σ e^(x−m)` and
  `t = Σ e^(x−m)·x`, giving `entropy = m + log s − t/s` (TRL's entropy) and/or `lse − x[target]`
  (Soup's per-row loss), in fp32. With these, accelerate's fp32 conversion of the model's outputs
  can skip the logits (every other output is still converted). TRL's accuracy is an argmax, which
  is the same on fp16 and on its exact fp32 copy.

### Ideas and sources (all kernel code is our own)

- NF4 format and memory layout: bitsandbytes (MIT), `csrc/kernels.cu` and `csrc/ops.cu` of 0.50.2;
  QLoRA, Dettmers et al. 2023.
- RMSNorm: Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"; one-kernel norms as in
  the [Triton layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html);
  fusing the residual add into the norm is a common inference-engine fusion.
- RoPE: Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding".
- Online softmax: Milakov & Gimelshein 2018, "Online normalizer calculation for softmax".
- Computing the loss gradient inside a fused loss kernel, and avoiding fp32 logits: the idea
  behind Liger Kernel's fused cross-entropy (Hsu et al. 2024) and Apple's Cut Cross-Entropy
  (Wijmans et al. 2024). We used neither library (both are banned) and none of their code.
- The SwiGLU kernel in `solution/swiglu.py` is the organizers' example, unchanged.
- Code reading for the patch points: [Soup](https://github.com/MakazhanAlpamys/Soup) v0.77.0,
  transformers v5.16.1, TRL v0.29.1, accelerate v1.14.0, bitsandbytes 0.50.2.

## How to run

**On Kaggle (what the jury runs).** Open our public notebook (link above) and press
**Copy & Edit**, or import
`templates/kaggle_notebook.ipynb` into a new notebook. Settings: **Accelerator GPU T4 x2**,
**Internet on**. In the first cell set

```python
REPO_URL = "https://github.com/Iliyas1996/SoupKernel.git"
COMMIT = ""   # empty = the latest commit; or the final commit from SUBMISSION.md
```

and press **Run All** (about 12 minutes). The notebook installs the pinned packages, clones this
commit, prepares the data, runs our kernel tests (`pytest solution`), trains base Soup and our
solution, evaluates both adapters and prints the checks, ending with:

```
speedup 1.135x   memory base/solution 1.944x   eval loss vs base -0.000 %
ALL CHECKS PASSED
```

Nothing else is needed: the harness calls `solution.apply()`, and the solution has no settings.

**Locally** (Linux, CUDA GPU, Python 3.13):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q          # our kernels against the code they replace
python harness/run_task.py            # base and solution, writes runs/result.json
python harness/check_rules.py && python harness/score.py runs/result.json
```

**For analysis only** (not part of the measured runs):

```bash
python harness/run_task.py --only solution --out runs_try --profile    # per-kernel GPU time
SOUP_KERNEL_DIAG=1 python harness/run_task.py --only solution --out runs_diag
grep "\[diag\]" runs_diag/solution_1/train.log                        # dtypes + memory timeline
```

`SOUP_KERNEL_DIAG` is off by default. When on, it only prints dtypes and
`torch.cuda.memory_allocated` at points of one step; it changes no computation.

## Results

From our Kaggle run of commit `900e471` (one base and one solution run; the jury runs two each):

| Check | Result |
|---|---|
| 1. all runs finished | PASS |
| 2. no ready-made kernel libraries | PASS |
| 3. own kernels run in training | PASS: all 7 kernels above |
| 4. own kernels ≥ 5% of training time | PASS: 10.0% |
| 5. same task and Soup outputs (tokens, metrics, rewind) | PASS |
| 6. loss curve | PASS: first step 0.02%, mean 0.01% |
| 7. eval loss | PASS: −0.000% |
| 8. not slower, not more memory | PASS: time ×0.881, memory ×0.515 |
| 9. clean measurement | PASS |
| 10. Soup's 79 regression tests | PASS |

How we got there (Kaggle T4, one run each; the base took 75–78 s and 14,345 MiB in every run):

| Version | Time | Peak MiB | Memory ratio | Own-kernel share |
|---|---|---|---|---|
| Starter kit (example SwiGLU only) | 78.8 s | 12,385 | 1.158 | 0% |
| + NF4 dequant + RMSNorm | 76.6 s | 12,383 | 1.158 | about 5.4% (from the profile: too close to 5%) |
| + RoPE + fused add/RMSNorm | 74.4 s | 12,381 | 1.159 | 7.1% |
| + fused cross-entropy + fp16 MLP intermediates | 70.0 s | 9,161 | 1.566 | 8.8% |
| + metrics from fp16 logits (final) | 67.6 s | 7,381 | 1.944 | 10.0% |

## Limitations and what did not work

- **The speedup is modest (1.13x).** The matrix multiplies (about 36% of the GPU time) still run in
  cuBLAS after a separate dequantization. A fused NF4 dequant + matmul kernel in Triton is the next
  big step, but on a T4 it has to beat cuBLAS to pay off, and we did not get to it.
- **Our first kernels saved no memory.** The fused RMSNorm alone left the peak unchanged, because
  with fp32 inputs PyTorch's norm kept little extra. Only the memory timeline showed us where the
  peak really was: the MLP activations and the copies of the logits.
- **The fp32 residual stream is still there.** About 18% of the GPU time is still dtype casts
  around the 4-bit layers and LoRA. Writing the norms' outputs directly in fp16, and fusing the
  LoRA branch (dropout, two small matmuls, scale and add), would remove most of them.
- **What still takes memory at the peak:** the MLP intermediates in fp16 (about 2 GB at 2,048
  tokens), the fp16 copy of the tied lm_head weight that autograd keeps (0.45 GB), the fp16 logits
  (0.6 GB), and a transient fp16 copy of them in TRL's accuracy metric (0.6 GB). Recomputing in
  the backward pass would cut more, but that is close to gradient checkpointing, which must stay
  off.
- **The logged metrics are computed differently, with the same math.** TRL's entropy and Soup's
  per-row losses now come from our kernel, in fp32, from the fp16 logits (the fp32 copy they used
  held exactly the same values); check 5 confirms they match the base.
- **Version-pinned patches.** We patch internals of the pinned versions: bitsandbytes 0.50.2
  (`backends.cuda.ops._dequantize_4bit_impl`), transformers 5.16.1 (Qwen2 forward functions,
  `LOSS_MAPPING`), accelerate 1.14 (`ConvertOutputsToFp32`), TRL 0.29.1 (`entropy_from_logits`)
  and Soup 0.77.0 (`rewind_hf.row_losses`). Every patch falls back to the original code for inputs
  it does not handle.
- **Measurement noise.** Our numbers are single runs; the organizers report a 1–4% spread between
  repeats on the T4.

## Repository layout

```
solution/__init__.py          apply(): turns on all patches
solution/nf4_dequant.py       NF4 dequantization kernel (bitsandbytes)
solution/rmsnorm.py           RMSNorm and fused add + RMSNorm kernels (Qwen2)
solution/rope.py              rotary embedding kernel (Qwen2 attention)
solution/cross_entropy.py     fused causal-LM loss kernels (transformers)
solution/logits_metrics.py    entropy / per-token loss kernel (TRL, Soup, accelerate)
solution/mlp.py               fp16 MLP intermediates around the example SwiGLU
solution/swiglu.py            the organizers' example kernel (unchanged)
solution/diagnostics.py       opt-in dtype and memory timeline (SOUP_KERNEL_DIAG=1)
solution/test_own_kernels.py  GPU tests of our kernels against the code they replace
solution/test_kernels.py      the example's test
harness/, task/, tests/       the organizers' measurement scripts and Soup tests (unchanged)
templates/                    the organizers' Kaggle notebook and submission template
```

The starter kit's original README (task, rules and harness description) is in the organizers'
repository: [MakazhanAlpamys/soup-kernel-challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge).
