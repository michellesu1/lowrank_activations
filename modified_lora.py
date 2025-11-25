# llama_svd_mem_fused.py
# pip install torch transformers accelerate wandb

import argparse, time, math
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import LlamaForCausalLM, AutoTokenizer
import wandb

import sys
sys.stdout = open("debug_output.txt", "w")

# -------------------- memory helpers --------------------
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

# -------------------- tiny synthetic corpus --------------------
def make_toy_texts(num=4096):
    base = [
        "The quick brown fox jumps over the lazy dog.",
        "Large language models enable powerful text generation.",
        "Parameter-efficient fine-tuning reduces compute and memory.",
        "Activation memory dominates backpropagation costs.",
        "Low-rank projection can reduce GPU storage.",
        "Tiny datasets are handy for quick debugging.",
        "Edge devices benefit from compact models.",
    ]
    texts = (base * ((num + len(base) - 1) // len(base)))[:num]
    return texts

def toy_batch(tok, texts, device):
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    enc["labels"] = enc["input_ids"].clone()
    return enc

# -------------------- Fused custom linear --------------------
class LowRankLinearFunction(torch.autograd.Function):
    total_forward_calls = 0
    total_backward_calls = 0

    @staticmethod
    def forward(ctx, x, W_eff, b, Vk):
        LowRankLinearFunction.total_forward_calls += 1
        # Print every 100 calls to avoid log spam
        # if LowRankLinearFunction.total_forward_calls % 100 == 0:
            # print(f"[LowRankLinearFunction] FORWARD call {LowRankLinearFunction.total_forward_calls}")
        Z = x @ Vk         # [*, k]
        WV = W_eff @ Vk    # [out, k]
        y = Z @ WV.t()     # [*, out]
        if b is not None:
            y = y + b
        ctx.save_for_backward(Z, WV, Vk)
        print(f"[DEBUG] LowRankLinearFunction.forward ctx save: Z.shape={Z.shape}, WV.shape={WV.shape}, Vk.shape={Vk.shape}")
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def backward(ctx, grad_output):
        LowRankLinearFunction.total_backward_calls += 1
        # if LowRankLinearFunction.total_backward_calls % 100 == 0:
        #     # print(f"[LowRankLinearFunction] BACKWARD call {LowRankLinearFunction.total_backward_calls}")
        Z, WV, Vk = ctx.saved_tensors
        print(f"[DEBUG] LowRankLinearFunction.backward ctx restore: Z.shape={Z.shape}, WV.shape={WV.shape}, Vk.shape={Vk.shape}")
        grad_Z = grad_output @ WV                            # [*, k]
        grad_WV = grad_output.transpose(-1, -2).reshape(-1, grad_output.shape[-1]).t() @ Z.reshape(-1, Z.shape[-1])  # [out, k]
        grad_x = grad_Z @ Vk.t()                             # [*, d_in]
        grad_W_eff = grad_WV @ Vk.t()                        # [out, d_in]
        grad_b = grad_output.sum(tuple(range(grad_output.ndim - 1))) if ctx.has_bias else None
        grad_Vk = None  # Typically basis is frozen after warmup
        return grad_x, grad_W_eff, grad_b, grad_Vk

class LowRankFusedModule(nn.Module):
    def __init__(self, module: nn.Module, Vk: torch.Tensor):
        super().__init__()
        self.mod = module
        self.register_buffer("Vk", Vk.cpu().float(), persistent=False)

    def _effective_Wb_oriented(self, x):
        # Base: always use the (swapped) module's .weight + .bias.
        base = self.mod
        if not hasattr(base, "weight"):
            raise RuntimeError(f"Base module has no weight: {type(base)}")
        W_base = base.weight
        b      = getattr(base, "bias", None)
        d_in = x.shape[-1]
        # Orient to [out, d_in]
        if W_base.ndim == 2 and W_base.shape[1] == d_in:
            W_eff = W_base
        elif W_base.ndim == 2 and W_base.shape[0] == d_in:
            W_eff = W_base.t().contiguous()
        else:
            if d_in == W_base.shape[0]:
                W_eff = W_base.t().contiguous()
            elif d_in == W_base.shape[1]:
                W_eff = W_base
            else:
                raise RuntimeError(f"Cannot orient W: {tuple(W_base.shape)} vs input dim {d_in}")
        return W_eff, b, base

    def forward(self, x):
        Vk = self.Vk.to(x.device, x.dtype)
        # print("x.shape", x.shape, "Vk.shape", Vk.shape)
        W_eff, b, _ = self._effective_Wb_oriented(x)
        # print(f"[LowRankFusedModule] Calling forward")
        return LowRankLinearFunction.apply(x, W_eff, b, Vk)

# -------------------- basis collector + in-place swapper --------------------
class BasisCollectorAndSwapper:
    """
    Warm-up to build Vk at selected modules, then swap each with LowRankFusedModule(Vk).
    After swap: fused op does activation projection for memory savings.
    """
    def __init__(self, rank=32, collect_steps=30, sample_rows=16, method="pca",
                 targets=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj")):
        self.rank = rank
        self.collect_steps = collect_steps
        self.sample_rows = sample_rows
        self.method = method  # "pca" or "svd"
        self.targets = targets
        self.state = {}       # name -> {"Vk": None|Tensor, "buf": [], "count": 0}
        self.root = None

    def _want(self, name, module):
        forbidden_suffixes = (
            ".loraA", ".loraB", ".lora_dropout", ".loraembeddingA", ".loraembeddingB", ".loramagnitudevector"
        )
        if "base_layer" in name or any(name.endswith(suf) for suf in forbidden_suffixes):
            return False
        return any(t in name for t in self.targets)

    def attach(self, model):
        self.root = model
        handles = []
        for name, mod in model.named_modules():
            if self._want(name, mod):
                self.state[name] = {"Vk": None, "buf": [], "count": 0}
                handles.append(mod.register_forward_pre_hook(self._pre(name), with_kwargs=False))
                handles.append(mod.register_forward_hook(self._post(name), with_kwargs=False))
        return handles

    def _pre(self, name):
        def hook(module, args):
            (x,) = args
            st = self.state[name]
            if st["Vk"] is None and st["count"] < self.collect_steps:
                with torch.no_grad():
                    x2d = x.reshape(-1, x.shape[-1])
                    B = x2d.shape[0]
                    st["buf"].append(x2d[:B].detach().cpu().float())
                    st["count"] += 1
                return
        return hook

    def _post(self, name):
        def hook(module, args, out):
            st = self.state[name]
            if st["Vk"] is None and st["count"] >= self.collect_steps and st["buf"]:
                with torch.no_grad():
                    X = torch.cat(st["buf"], dim=0)  # [N, d]
                    d = X.shape[-1]
                    k = min(self.rank, d, X.shape[0])
                    # print(f"ACTIVATION BUFF SHAPE: {[a.shape for a in st['buf']]}")
                    # print(f"X SHAPE (concat): {X.shape}")
                    try:
                        if self.method == "pca":
                            U, S, V = torch.pca_lowrank(X, q=k, center=False)
                            Vk = V[:, :k].contiguous()
                        else:
                            U_, S_, Vh = torch.svd_lowrank(X, q=k)
                            Vk = Vh[:k, :].t().contiguous()
                    except Exception:
                        U_, S_, Vh = torch.svd_lowrank(X, q=k)
                        Vk = Vh[:k, :].t().contiguous()
                    del X
                    if "U" in locals(): del U
                    if "S" in locals(): del S
                    if "V" in locals(): del V
                    if "U_" in locals(): del U_
                    if "S_" in locals(): del S_
                    if "Vh" in locals(): del Vh
                    st["Vk"] = Vk
                    st["buf"].clear()
                    self._replace_by_name(self.root, name, LowRankFusedModule(module, Vk))
                    # print("Vk shape: ", Vk.shape)
        return hook

    @staticmethod
    def _replace_by_name(root, dotted, new_mod):
        forbidden = ("lora_A", "lora_B", "lora_dropout")
        if any(f in dotted for f in forbidden):
            return
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

# -------------------- LR scheduler helper --------------------
def make_scheduler(optimizer, total_steps, warmup_steps=0, kind="cosine"):
    if kind == "none":
        return None
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if kind == "linear":
            return 1.0 - progress
        elif kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------- training loop --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--svd_rank", type=int, default=32)
    ap.add_argument("--svd_collect", type=int, default=30)
    ap.add_argument("--svd_sample_rows", type=int, default=16)
    ap.add_argument("--svd_method", choices=["pca","svd"], default="pca")
    ap.add_argument("--targets", default="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj",
                    help="comma-separated module-name substrings")
    ap.add_argument("--save_dir", default=None, help="Directory to save model checkpoint")
    # W&B + LR schedule
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="llama-svd-fused", help="W&B project name")
    ap.add_argument("--wandb_run_name", default=None, help="W&B run name (optional)")
    ap.add_argument("--lr_schedule", choices=["none", "cosine", "linear"], default="cosine",
                    help="Learning-rate schedule")
    ap.add_argument("--lr_warmup_steps", type=int, default=50, help="Warmup steps for LR schedule")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = LlamaForCausalLM.from_pretrained(args.model).to(device)

    inserter = BasisCollectorAndSwapper(
        rank=args.svd_rank,
        collect_steps=args.svd_collect,
        sample_rows=args.svd_sample_rows,
        method=args.svd_method,
        targets=tuple(t.strip() for t in args.targets.split(",")),
    )
    handles = inserter.attach(model)

    # Diagnostic: Print all torch.nn.Linear modules still present
    print("=== All linear modules after swapping ===")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Linear: {name}")

    # Diagnostic: Print all swapped modules (should match your targets)
    print("=== All LowRankFusedModules after swapping ===")
    for name, module in model.named_modules():
        if isinstance(module, LowRankFusedModule):
            print(f"Swapped: {name}")

    # ----- W&B init -----
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_schedule": args.lr_schedule,
                "lr_warmup_steps": args.lr_warmup_steps,
                "svd_rank": args.svd_rank,
                "svd_collect": args.svd_collect,
                "svd_sample_rows": args.svd_sample_rows,
                "svd_method": args.svd_method,
                "targets": args.targets,
            }
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.watch(model, log="gradients", log_freq=50)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    texts = make_toy_texts(10000)

    total_steps = min(args.steps, len(texts) // args.batch_size)
    scheduler = make_scheduler(
        opt, total_steps=total_steps,
        warmup_steps=args.lr_warmup_steps,
        kind=args.lr_schedule
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak()

    t0 = time.time(); token_count = 0
    in_project_mode = False
    steady_peak = 0.0
    overall_peak = 0.0

    for step in range(total_steps):
        s, e = step*args.batch_size, (step+1)*args.batch_size
        batch = toy_batch(tok, texts[s:e], device)

        out = model(**batch)
        loss = out.loss
        loss.backward()
        mem_stats = get_mem_mb()
        print(f"After backward: CUDA alloc={mem_stats['alloc']:.1f}MB peak={mem_stats['peak']:.1f}MB reserved={mem_stats['reserved']:.1f}MB")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

        opt.step()
        mem_stats = get_mem_mb()
        print(f"After optimizer: CUDA alloc={mem_stats['alloc']:.1f}MB peak={mem_stats['peak']:.1f}MB reserved={mem_stats['reserved']:.1f}MB")
        if scheduler is not None:
            scheduler.step()
        opt.zero_grad(set_to_none=True)

        token_count += int(batch["input_ids"].numel())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            overall_peak = max(overall_peak, torch.cuda.max_memory_allocated()/1e6)

        mode_collecting = any(st["Vk"] is None for st in inserter.state.values())
        if not in_project_mode and not mode_collecting:
            print(f"==> All targets swapped to fused low-rank at step {step}. Resetting peak.")
            reset_peak()
            in_project_mode = True
            mem = get_mem_mb()
            print(
                f"[SWAP] Post-swap memory: alloc={mem['alloc']:.1f}MB, peak={mem['peak']:.1f}MB, reserved={mem['reserved']:.1f}MB"
            )

            print("=== Per-module Vk buffer reports after swap ===")
            total_swapped = 0
            for name, module in model.named_modules():
                if isinstance(module, LowRankFusedModule):
                    print("Swapped:", name)
                    total_swapped += 1
                    buf = getattr(module, "Vk", None)
                    if buf is not None:
                        print(f"  {name}: Vk.shape={tuple(buf.shape)}, Vk.device={buf.device}, Vk.dtype={buf.dtype}, Vk.nbytes={buf.element_size() * buf.nelement() / 1e6:.1f}MB")
            print("===============================================")
            print(f"Total swapped: {total_swapped}")

            print("=== Remaining standard linear modules after all swaps ===")
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    print("Linear:", name)

        if in_project_mode and torch.cuda.is_available():
            steady_peak = max(steady_peak, torch.cuda.max_memory_allocated()/1e6)

        if step % 10 == 0:
            mem = get_mem_mb()
            mode = "collect" if mode_collecting else "project"
            current_lr = opt.param_groups[0]["lr"]
            print(f"[fused] step {step:03d} | loss {loss.item():.3f} | mode={mode} | "
                  f"lr={current_lr:.6g} | grad_norm={float(grad_norm):.3f} | "
                  f"alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")

        if args.wandb:
            mem_now = get_mem_mb()
            current_lr = opt.param_groups[0]["lr"]
            wandb.log({
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/lr": float(current_lr),
                "train/grad_norm": float(grad_norm),
                "train/mem_alloc_MB": mem_now["alloc"],
                "train/mem_peak_MB": mem_now["peak"],
                "train/mem_reserved_MB": mem_now["reserved"],
                "train/mode_collect": 1.0 if mode_collecting else 0.0,
                "train/tokens_cum": token_count,
            })

    print("=== Collector State Summary ===")
    print("Total FORWARD calls:", LowRankLinearFunction.total_forward_calls)
    print("Total BACKWARD calls:", LowRankLinearFunction.total_backward_calls)
    for name, st in inserter.state.items():
        print(f"{name}: collected {st['count']} / {inserter.collect_steps}, Vk: {'set' if st['Vk'] is not None else 'None'}")

    dt = time.time() - t0
    mem = get_mem_mb()
    print("="*80)
    print(f"Llama-3 + Fused Low-Rank (targets={args.targets})")
    print(f"model={args.model} steps={total_steps} batch={args.batch_size} "
          f"rank={args.svd_rank} collect={args.svd_collect} method={args.svd_method}")
    print(f"tokens={token_count} throughput={token_count/max(dt,1e-6):.1f} toks/s")
    print(f"CUDA alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")
    print(f"OVERALL PEAK: {overall_peak:.1f}MB")
    if in_project_mode:
        print(f"STEADY-STATE PEAK (after swap): {steady_peak:.1f}MB")
    print("="*80)

    if args.save_dir is not None:
        print(f"Saving model checkpoint to {args.save_dir} ...")
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"Checkpoint saved at {args.save_dir}")

    for h in handles: h.remove()

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
