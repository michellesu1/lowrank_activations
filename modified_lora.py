# lora_svd_mem_fused.py
# pip install torch transformers peft accelerate

import argparse, time
import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

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

# -------------------- LoRA helper --------------------
def add_lora(model, r=8, alpha=16, dropout=0.05, target_modules=None):
    target_modules = target_modules or ["c_attn", "c_proj"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

# -------------------- fused low-rank autograd --------------------
class LowRankFusedModule(nn.Module):
    def __init__(self, module: nn.Module, Vk: torch.Tensor):
        super().__init__()
        self.mod = module            # may be PEFT LoRA wrapper
        self.register_buffer("Vk", Vk.float(), persistent=False)

    @staticmethod
    def _peel_wrappers(mod):
        """
        Returns (outer_wrapper, base) where:
          - outer_wrapper is the first module that may carry LoRA params (lora_A/B, scaling, fan_in_fan_out)
          - base is the leaf module that actually owns .weight/.bias (Linear or Conv1D)
        This also peels off our own LowRankFusedModule if it appears as a base.
        """
        outer = mod
        base  = mod

        # keep the first module that has LoRA attrs as 'outer'
        if not (hasattr(outer, "lora_A") and hasattr(outer, "lora_B")):
            outer = None

        # unwrap repeatedly: PEFT wrappers have .base_layer
        seen = set()
        cur = mod
        while True:
            if id(cur) in seen:
                break  # safety
            seen.add(id(cur))

            # If it's our own wrapper (from an earlier swap), peel to its .mod
            if isinstance(cur, LowRankFusedModule):
                cur = cur.mod
                continue

            # If it exposes LoRA attrs and we haven't recorded an outer yet, set it
            if outer is None and hasattr(cur, "lora_A") and hasattr(cur, "lora_B"):
                outer = cur

            # PEFT LoRA wrappers have .base_layer
            if hasattr(cur, "base_layer"):
                cur = cur.base_layer
                continue
            break

        base = cur
        return outer, base

    @staticmethod
    def _is_conv1d_like(base):
        return hasattr(base, "weight") and base.weight.ndim == 2 and not isinstance(base, nn.Linear)

    @staticmethod
    def _get_lora_tuple(wrapper):
        # ... your usual logic ...
        if hasattr(wrapper, "lora_A") and hasattr(wrapper, "lora_B") and len(wrapper.lora_A) > 0:
            name = next(iter(wrapper.lora_A.keys()))
            A = wrapper.lora_A[name]
            B = wrapper.lora_B[name]
            # Insert here:
            if isinstance(A, nn.Linear):
                A = A.weight
            if isinstance(B, nn.Linear):
                B = B.weight
            scaling = float(wrapper.scaling.get(name, wrapper.scaling.get("default", 1.0)))
            fio = bool(getattr(wrapper, "fan_in_fan_out", False))
            return (A, B, scaling, fio)
        return (None, None, 1.0, False)


    def _effective_Wb_oriented(self, x):
        # unwrap chain
        wrapper, base = self._peel_wrappers(self.mod)

        if not hasattr(base, "weight"):
            raise RuntimeError(f"Base module has no weight: {type(base)}")
        W_base = base.weight
        b      = getattr(base, "bias", None)

        # LoRA delta from the outermost wrapper (if present)
        A, B, scaling, fio = self._get_lora_tuple(wrapper)
        # print("A shape:", A.shape)
        # print("B shape:", B.shape)
        # print("Delta shape:", (B @ A).shape)
        if A is not None:
            if fio:
                # fan-in/fan-out path (Conv1D-style): W <- W + (A @ B).T * scaling
                delta = (B @ A).t()
            else:
                # Linear-style: W <- W + (B @ A) * scaling
                delta = (B @ A)

            if delta.shape == W_base.shape:
                W_eff_raw = W_base + delta * scaling
            elif delta.t().shape == W_base.shape:
                W_eff_raw = W_base + delta.t() * scaling
            else:
                # unexpected; fall back to base only
                W_eff_raw = W_base
        else:
            W_eff_raw = W_base

        d_in = x.shape[-1]

        # Orient to [out, d_in]
        if W_eff_raw.ndim == 2 and W_eff_raw.shape[1] == d_in:
            W_eff = W_eff_raw
        elif W_eff_raw.ndim == 2 and W_eff_raw.shape[0] == d_in:
            W_eff = W_eff_raw.t().contiguous()
        else:
            if d_in == W_eff_raw.shape[0]:
                W_eff = W_eff_raw.t().contiguous()
            elif d_in == W_eff_raw.shape[1]:
                W_eff = W_eff_raw
            else:
                raise RuntimeError(f"Cannot orient W: {tuple(W_eff_raw.shape)} vs input dim {d_in}")

        return W_eff, b, base  # base is where grads must land

    def forward(self, x):
        Vk = self.Vk.to(x.device, x.dtype)
        W_eff, b, base = self._effective_Wb_oriented(x)

        Z  = x.matmul(Vk)                 # [..., k]
        WV = W_eff.matmul(Vk)             # [out, k]
        y  = Z.matmul(WV.t().to(x.dtype))
        if b is not None:
            y = y + b.to(y.dtype)

        if self.training:
            Zc, WVc, Vkc = Z.detach(), WV.detach(), Vk.detach()
            W_param = base.weight
            b_param = getattr(base, "bias", None)

            def _hook(gy):
                gx  = gy.matmul(WVc).matmul(Vkc.t())
                gWV = gy.transpose(-1,-2).reshape(-1, gy.shape[-1]).t().matmul(Zc.reshape(-1, Zc.shape[-1]))  # [out,k]
                gW  = gWV.matmul(Vkc.t())  # [out, d]
                # map to base weight layout
                if W_param.shape == gW.shape:
                    gW_base = gW
                elif W_param.shape == gW.t().shape:
                    gW_base = gW.t()
                else:
                    raise RuntimeError(f"Grad shape mismatch: W {tuple(W_param.shape)} vs gW {tuple(gW.shape)}")
                if W_param.grad is None: W_param.grad = torch.zeros_like(W_param)
                W_param.grad = W_param.grad + gW_base
                if b_param is not None:
                    gb = gy.reshape(-1, gy.shape[-1]).sum(0).to(b_param.dtype)
                    if b_param.grad is None: b_param.grad = torch.zeros_like(b_param)
                    b_param.grad = b_param.grad + gb
                return gx

            y.register_hook(_hook)

        return y


# -------------------- basis collector + in-place swapper --------------------
class BasisCollectorAndSwapper:
    """
    Warm-up to build Vk at selected modules, then swap each with LowRankFusedModule(Vk).
    After swap: no pre-projection; fused op enforces low-rank and saves k-dim state.
    """
    def __init__(self, rank=32, collect_steps=30, sample_rows=16, method="pca",
                 targets=("attn.c_attn","attn.c_proj")):
        self.rank = rank
        self.collect_steps = collect_steps
        self.sample_rows = sample_rows
        self.method = method  # "pca" or "svd"
        self.targets = targets
        self.state = {}       # name -> {"Vk": None|Tensor, "buf": [], "count": 0}
        self.root = None

    # in BasisCollectorAndSwapper._want / want
    def _want(self, name, module):
        # match targets by name, but avoid internal base_layer nodes
        if "base_layer" in name:
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
                    B = min(self.sample_rows, x2d.shape[0])
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
                    d = X.shape[-1]; k = min(self.rank, d)
                    try:
                        if self.method == "pca":
                            U, S, V = torch.pca_lowrank(X, q=k, center=False)
                        else:
                            U_, S_, Vh = torch.svd_lowrank(X, q=k)
                            V = Vh.t()
                    except Exception:
                        U_, S_, Vh = torch.svd_lowrank(X, q=k)
                        V = Vh.t()
                    Vk = V[:, :k].contiguous()        # CPU fp32
                    st["Vk"] = Vk
                    st["buf"].clear()
                    # ---- swap this module with fused low-rank wrapper ----
                    self._replace_by_name(self.root, name, LowRankFusedModule(module, Vk))
        return hook

    @staticmethod
    def _replace_by_name(root, dotted, new_mod):
        # Only replace top-level modules, not LoRA's internal attributes
        forbidden = ("lora_A", "lora_B", "lora_dropout")
        if any(f in dotted for f in forbidden):
            # print(f"SKIP REPLACING {dotted}: is a LoRA param or submodule")
            return
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        # print(f"Replacing {dotted}: was {type(getattr(parent, parts[-1]))}, now {type(new_mod)})")
        setattr(parent, parts[-1], new_mod)


# -------------------- training loop --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--svd_rank", type=int, default=32)
    ap.add_argument("--svd_collect", type=int, default=30)
    ap.add_argument("--svd_sample_rows", type=int, default=16)
    ap.add_argument("--svd_method", choices=["pca","svd"], default="pca")
    ap.add_argument("--targets", default="attn.c_attn,attn.c_proj",
                    help="comma-separated module-name substrings; add mlp.c_fc if desired")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # LoRA (weights low-rank) unchanged
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout, target_modules=["c_attn","c_proj"])

    # Learn Vk, then swap to fused low-rank layers at targets
    inserter = BasisCollectorAndSwapper(
        rank=args.svd_rank,
        collect_steps=args.svd_collect,
        sample_rows=args.svd_sample_rows,
        method=args.svd_method,
        targets=tuple(t.strip() for t in args.targets.split(",")),
    )
    handles = inserter.attach(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    texts = make_toy_texts(10000)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak()

    t0 = time.time(); token_count = 0
    in_project_mode = False
    steady_peak = 0.0
    overall_peak = 0.0

    for step in range(args.steps):
        s, e = step*args.batch_size, (step+1)*args.batch_size
        if e > len(texts): break
        batch = toy_batch(tok, texts[s:e], device)

        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)

        token_count += int(batch["input_ids"].numel())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            overall_peak = max(overall_peak, torch.cuda.max_memory_allocated()/1e6)

        # detect when all targets have Vk and were swapped
        mode_collecting = any(st["Vk"] is None for st in inserter.state.values())
        if not in_project_mode and not mode_collecting:
            print(f"==> All targets swapped to fused low-rank at step {step}. Resetting peak.")
            reset_peak()
            in_project_mode = True

        if in_project_mode and torch.cuda.is_available():
            steady_peak = max(steady_peak, torch.cuda.max_memory_allocated()/1e6)

        if step % 10 == 0:
            mem = get_mem_mb()
            mode = "collect" if mode_collecting else "project"
            print(f"[fused] step {step:03d} | loss {loss.item():.3f} | mode={mode} | "
                  f"alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")

    dt = time.time() - t0
    mem = get_mem_mb()
    print("="*80)
    print(f"LoRA + Fused Low-Rank (targets={args.targets})")
    print(f"model={args.model} steps={args.steps} batch={args.batch_size} "
          f"rank={args.svd_rank} collect={args.svd_collect} method={args.svd_method}")
    print(f"tokens={token_count} throughput={token_count/max(dt,1e-6):.1f} toks/s")
    print(f"CUDA alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")
    print(f"OVERALL PEAK: {overall_peak:.1f}MB")
    if in_project_mode:
        print(f"STEADY-STATE PEAK (after swap): {steady_peak:.1f}MB")
    print("="*80)

    for h in handles: h.remove()

if __name__ == "__main__":
    main()
