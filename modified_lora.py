# lora_svd_mem.py
# pip install torch transformers peft accelerate

import argparse, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ---- memory helpers ----
def get_mem_mb():
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "peak": 0.0, "reserved": 0.0}
    return {
        "alloc": torch.cuda.memory_allocated() / 1e6,
        "peak": torch.cuda.max_memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
    }

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

# ---- tiny synthetic corpus ----
def make_toy_texts(num=4096):
    base = [
        "The quick brown fox jumps over the lazy dog.",
        "Large language models enable powerful text generation.",
        "Parameter-efficient fine-tuning reduces compute and memory.",
        "Activation memory dominates backpropagation costs.",
        "LoRA adds low-rank adapters to attention projections.",
        "Tiny datasets are handy for quick debugging.",
        "Edge devices benefit from compact models.",
    ]
    texts = (base * ((num + len(base) - 1) // len(base)))[:num]
    return texts

def toy_batch(tok, texts, device):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    enc["labels"] = enc["input_ids"].clone()
    return enc

# ---- LoRA helper ----
def add_lora(model, r=8, alpha=16, dropout=0.05, target_modules=None):
    target_modules = target_modules or ["c_attn", "c_proj"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

# ---- Activation projector hooks with truncated/randomized SVD ----
class FrozenFeatureProjector:
    """
    Per-module projector:
      - Collect tiny input slices for N steps (CPU fp32 buffer).
      - Build Vk once using torch.pca_lowrank (randomized PCA) or torch.svd_lowrank (truncated SVD).
      - Thereafter, project inputs: x -> x @ Vk @ Vk^T (along last dim).
    """
    def __init__(self, rank=32, collect_steps=50, sample_rows=16, method="pca"):
        self.rank = rank
        self.collect_steps = collect_steps
        self.sample_rows = sample_rows
        self.method = method  # "pca" or "svd"
        self.state = {}       # name -> {"Vk": Tensor|None, "samples": list, "count": int}

    def want(self, name, module):
        # Target GPT-2 style attention linears (adjust as needed)
        return isinstance(module, nn.Linear) and ("attn.c_attn" in name or "attn.c_proj" in name)

    def pre_hook(self, name):
        def _pre(module, args):
            (x,) = args  # (..., d)
            st = self.state[name]
            if st["Vk"] is None and st["count"] < self.collect_steps:
                with torch.no_grad():
                    x2d = x.reshape(-1, x.shape[-1])
                    B = min(self.sample_rows, x2d.shape[0])
                    st["samples"].append(x2d[:B].detach().cpu().float())
                    st["count"] += 1
                return  # pass-through
            if st["Vk"] is not None:
                Vk = st["Vk"].to(x.device, x.dtype)  # (d, k)
                return ( (x @ Vk) @ Vk.t(), )
        return _pre

    def post_hook(self, name):
        def _post(module, args, out):
            st = self.state[name]
            if st["Vk"] is None and st["count"] >= self.collect_steps and st["samples"]:
                with torch.no_grad():
                    X = torch.cat(st["samples"], dim=0)  # [N, d]
                    d = X.shape[-1]
                    k = min(self.rank, d)
                    try:
                        if self.method == "pca":
                            # randomized PCA; faster on tall matrices
                            U, S, V = torch.pca_lowrank(X, q=k, center=False)
                        else:
                            # truncated SVD; randomized internally
                            U_, S_, Vh = torch.svd_lowrank(X, q=k)
                            V = Vh.t()
                    except Exception:
                        # fallback between methods if one fails
                        U_, S_, Vh = torch.svd_lowrank(X, q=k)
                        V = Vh.t()
                    st["Vk"] = V[:, :k].contiguous()    # store CPU fp32
                    st["samples"].clear()
        return _post

    def attach(self, model):
        handles = []
        for name, mod in model.named_modules():
            if self.want(name, mod):
                self.state[name] = {"Vk": None, "samples": [], "count": 0}
                handles.append(mod.register_forward_pre_hook(self.pre_hook(name), with_kwargs=False))
                handles.append(mod.register_forward_hook(self.post_hook(name), with_kwargs=False))
        return handles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--svd_rank", type=int, default=32)
    ap.add_argument("--svd_collect", type=int, default=30)
    ap.add_argument("--svd_sample_rows", type=int, default=16)
    ap.add_argument("--svd_method", choices=["pca","svd"], default="pca",
                    help="pca: torch.pca_lowrank (recommended); svd: torch.svd_lowrank")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout, target_modules=["c_attn","c_proj"])

    projector = FrozenFeatureProjector(rank=args.svd_rank,
                                       collect_steps=args.svd_collect,
                                       sample_rows=args.svd_sample_rows,
                                       method=args.svd_method)
    handles = projector.attach(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    texts = make_toy_texts(10000)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak()

    t0 = time.time(); token_count = 0
    in_project_mode = False
    steady_state_peak = 0.0
    overall_peak = 0.0

    for step in range(args.steps):
        s, e = step*args.batch_size, (step+1)*args.batch_size
        if e > len(texts): break
        b = toy_batch(tok, texts[s:e], device)

        out = model(**b)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

        token_count += int(b["input_ids"].numel())

        if torch.cuda.is_available(): torch.cuda.synchronize()
        mode = "collect" if any(st["Vk"] is None for st in projector.state.values()) else "project"

        # Track overall peak memory (never reset)
        overall_peak = max(overall_peak, torch.cuda.max_memory_allocated() / 1e6)

        # Detect transition from collect to project mode
        if not in_project_mode and mode == "project":
            print(f"==> Switching to projection mode at step {step}, resetting peak memory stats.")
            reset_peak()
            in_project_mode = True

        if in_project_mode:
            # Track steady-state peak (after reset)
            steady_state_peak = max(steady_state_peak, torch.cuda.max_memory_allocated() / 1e6)

        if step % 10 == 0:
            mem = get_mem_mb()
            print(f"[svd-{args.svd_method}] step {step:03d} | loss {loss.item():.3f} | mode={mode} | "
                  f"alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")

    dt = time.time() - t0
    mem = get_mem_mb()
    print("="*80)
    print(f"MODIFIED (LoRA + activation projection via {args.svd_method})")
    print(f"model={args.model} steps={args.steps} batch={args.batch_size} svd_rank={args.svd_rank} collect={args.svd_collect}")
    print(f"tokens={token_count} throughput={token_count/max(dt,1e-6):.1f} toks/s")
    print(f"CUDA alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")
    print(f"OVERALL PEAK (all phases): {overall_peak:.1f}MB")
    if in_project_mode:
        print(f"STEADY-STATE PEAK (projection phase): {steady_state_peak:.1f}MB")
    print("="*80)

    for h in handles: h.remove()

if __name__ == "__main__":
    main()
