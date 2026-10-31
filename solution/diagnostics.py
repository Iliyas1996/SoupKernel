"""Diagnostics for our own analysis, OFF by default (the measured runs never turn it on).

Turn on with the environment variable SOUP_KERNEL_DIAG=1. For one training step (the 4th) it
prints, prefixed with "[diag]":
- the dtypes that reach the norm, the attention projections, the MLP and the loss;
- GPU memory (allocated, MiB) at the points of the step, and the step's peak,
so we can see what the memory peak is made of. It only reads tensors; it changes nothing.
"""

import os

DIAG_STEP = 4


def enabled():
    return os.environ.get("SOUP_KERNEL_DIAG") == "1"


def enable():
    import torch
    import transformers
    from transformers.models.qwen2 import modeling_qwen2

    state = {"step": 0, "seen": set()}

    def mib(n):
        return f"{n / 2**20:8.0f}"

    def active():
        return state["step"] == DIAG_STEP

    def mem(label):
        if active():
            torch.cuda.synchronize()
            print(f"[diag] mem {label:34s} alloc {mib(torch.cuda.memory_allocated())} MiB   "
                  f"peak-so-far {mib(torch.cuda.max_memory_allocated())} MiB", flush=True)

    def once(key, text):
        if active() and key not in state["seen"]:
            state["seen"].add(key)
            print(f"[diag] {key}: {text}", flush=True)

    # one step = one Trainer.training_step
    original_step = transformers.Trainer.training_step

    def training_step(self, *args, **kwargs):
        state["step"] += 1
        if active():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem("step start")
        out = original_step(self, *args, **kwargs)
        if active():
            mem("step end (after backward)")
        return out

    transformers.Trainer.training_step = training_step

    original_compute_loss = transformers.Trainer.compute_loss

    def compute_loss(self, model, inputs, *args, **kwargs):
        if active():
            ids = inputs.get("input_ids")
            once("batch", f"input_ids {tuple(ids.shape) if ids is not None else None}")
        out = original_compute_loss(self, model, inputs, *args, **kwargs)
        mem("after model forward + loss")
        return out

    transformers.Trainer.compute_loss = compute_loss

    try:  # TRL adds entropy and accuracy on top of the loss (and Soup the per-row losses)
        import trl

        original_sft_loss = trl.SFTTrainer.compute_loss

        def sft_compute_loss(self, *args, **kwargs):
            out = original_sft_loss(self, *args, **kwargs)
            mem("after TRL entropy + accuracy")
            return out

        trl.SFTTrainer.compute_loss = sft_compute_loss
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[diag] no TRL hook: {exc!r}", flush=True)

    original_model_forward = modeling_qwen2.Qwen2Model.forward

    def model_forward(self, *args, **kwargs):
        mem("before decoder layers")
        out = original_model_forward(self, *args, **kwargs)
        h = out.last_hidden_state
        once("final hidden", f"{h.dtype} {tuple(h.shape)}")
        mem("after decoder layers")
        return out

    modeling_qwen2.Qwen2Model.forward = model_forward

    original_lm_forward = modeling_qwen2.Qwen2ForCausalLM.forward

    def lm_forward(self, *args, **kwargs):
        out = original_lm_forward(self, *args, **kwargs)
        if active():
            logits = getattr(out, "logits", None)
            once("logits", f"{None if logits is None else (logits.dtype, tuple(logits.shape))}; "
                           f"lm_head weight {self.lm_head.weight.dtype}; "
                           f"autocast {torch.is_autocast_enabled('cuda')}")
            mem("after lm_head + model loss")
        return out

    modeling_qwen2.Qwen2ForCausalLM.forward = lm_forward

    original_mlp = modeling_qwen2.Qwen2MLP.forward

    def mlp_forward(self, x):
        if active() and "mlp" not in state["seen"]:
            gate = self.gate_proj
            once("mlp", f"input {x.dtype}; gate_proj {type(gate).__module__}.{type(gate).__name__} "
                        f"compute_dtype {getattr(gate, 'compute_dtype', None)} "
                        f"weight {gate.weight.dtype}")
        return original_mlp(self, x)

    modeling_qwen2.Qwen2MLP.forward = mlp_forward

    original_attn = modeling_qwen2.Qwen2Attention.forward

    def attn_forward(self, hidden_states, *args, **kwargs):
        if active() and "attention" not in state["seen"]:
            q = self.q_proj
            once("attention", f"input {hidden_states.dtype}; q_proj {type(q).__module__}."
                              f"{type(q).__name__}; k_proj {type(self.k_proj).__name__} "
                              f"compute_dtype {getattr(self.k_proj, 'compute_dtype', None)}")
        return original_attn(self, hidden_states, *args, **kwargs)

    modeling_qwen2.Qwen2Attention.forward = attn_forward

    original_norm = modeling_qwen2.Qwen2RMSNorm.forward

    def norm_forward(self, hidden_states):
        once("norm", f"input {hidden_states.dtype}; weight {self.weight.dtype}")
        return original_norm(self, hidden_states)

    modeling_qwen2.Qwen2RMSNorm.forward = norm_forward
    print("[diag] diagnostics on", flush=True)
