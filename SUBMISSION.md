# Submission: <TEAM NAME>

## Links
- Repository, final commit: https://github.com/Iliyas1996/SoupKernel/commit/<FINAL COMMIT SHA>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same notebook is
  linked in our README):
  <https://www.kaggle.com/code/ilyasasilgaly/notebook639bc5ef9a?scriptVersionId=<VERSION>>
- Demo video (up to 3 minutes): <VIDEO LINK>
- Presentation: <PRESENTATION LINK>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| <NAME> | <ROLE> | [Iliyas1996](https://github.com/Iliyas1996) | [ilyasasilgaly](https://www.kaggle.com/ilyasasilgaly) | <CONTACT> |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_unpack_scale_kernel` | `solution/nf4_dequant.py` | bitsandbytes NF4 dequantization (forward and backward of every 4-bit layer) | <SHARE>% |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm forward, and the residual add before `post_attention_layernorm` | <SHARE>% |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm backward, and the residual gradient add | <SHARE>% |
| `rope_rotate_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` for q and k, forward and backward | <SHARE>% |
| `ce_lse_loss_kernel` | `solution/cross_entropy.py` | transformers `ForCausalLMLoss` forward (fp32 logits copy + log_softmax) | <SHARE>% |
| `ce_grad_kernel` | `solution/cross_entropy.py` | the cross-entropy backward (writes the fp16 gradient directly) | <SHARE>% |
| `token_entropy_loss_kernel` | `solution/logits_metrics.py` | TRL `entropy_from_logits` and Soup rewind `row_losses` on fp32 logits | <SHARE>% |
| **Total** | | | **10.0%** |

Ideas taken from other work, with links (we wrote all kernel code ourselves):
- NF4 layout: [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 0.50.2
  `csrc/kernels.cu`; QLoRA (Dettmers et al. 2023, https://arxiv.org/abs/2305.14314).
- RMSNorm (https://arxiv.org/abs/1910.07467); one-kernel norms as in the
  [Triton layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).
- RoPE, RoFormer (https://arxiv.org/abs/2104.09864).
- Online softmax (https://arxiv.org/abs/1805.02867).
- Fused loss with the gradient computed in the loss kernel, no fp32 logits: Liger Kernel
  (https://arxiv.org/abs/2410.10989) and Cut Cross-Entropy (https://arxiv.org/abs/2411.09009);
  neither library nor their code is used.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 76.7 | 67.6 | speedup 1.135x |
| Peak GPU memory, MiB | 14,345 | 7,381 | base/ours 1.944x |
| Eval loss (public rows) | 2.04967 | 2.04966 | −0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (profile + memory timeline), how the kernels work, results
against the base (speed, memory, quality), limits and next steps.

## Limitations
See "Limitations and what did not work" in README.md. In short: the matrix multiplies (about 36%
of the GPU time) are untouched (no fused NF4 matmul yet); about 18% of the GPU time is still dtype
casts around the 4-bit layers and LoRA; our patches target the pinned library versions (each falls
back to the original code for inputs it does not handle); our numbers are single runs on a GPU
whose runs vary by 1–4%.
