# baseline_lora.py
# pip install torch transformers peft accelerate wandb

import argparse, time, math
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import wandb

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
    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

# ---- LR scheduler helper ----
def make_scheduler(optimizer, total_steps, warmup_steps=0, kind="cosine"):
    if kind == "none":
        return None
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if kind == "linear": return 1.0 - progress
        elif kind == "cosine": return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_dir", default=None, help="Directory to save final model checkpoint")
    # wandb + lr schedule
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="baseline-lora", help="W&B project name")
    ap.add_argument("--wandb_run_name", default=None, help="W&B run name (optional)")
    ap.add_argument("--lr_schedule", choices=["none", "cosine", "linear"], default="cosine", help="Learning-rate schedule")
    ap.add_argument("--lr_warmup_steps", type=int, default=50, help="Warmup steps for LR schedule")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = LlamaForCausalLM.from_pretrained(args.model).to(device)
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # W&B init
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
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            }
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.watch(model, log="gradients", log_freq=50)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    texts = make_toy_texts(10000)
    total_steps = min(args.steps, len(texts) // args.batch_size)
    scheduler = make_scheduler(opt, total_steps=total_steps,
                              warmup_steps=args.lr_warmup_steps,
                              kind=args.lr_schedule)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak()

    t0 = time.time(); token_count = 0
    for step in range(total_steps):
        s, e = step*args.batch_size, (step+1)*args.batch_size
        b = toy_batch(tok, texts[s:e], device)

        out = model(**b)
        loss = out.loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

        opt.step()
        if scheduler is not None: scheduler.step()
        opt.zero_grad(set_to_none=True)

        token_count += int(b["input_ids"].numel())
        if torch.cuda.is_available(): torch.cuda.synchronize()

        if step % 10 == 0:
            mem = get_mem_mb()
            lr_now = opt.param_groups[0]["lr"]
            print(f"[Llama3-Lora] step {step:03d} | loss {loss.item():.3f} | "
                  f"lr={lr_now:.6g} | grad_norm={float(grad_norm):.3f} | "
                  f"alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")

        if args.wandb:
            mem_now = get_mem_mb()
            lr_now = opt.param_groups[0]["lr"]
            wandb.log({
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/lr": float(lr_now),
                "train/grad_norm": float(grad_norm),
                "train/mem_alloc_MB": mem_now["alloc"],
                "train/mem_peak_MB": mem_now["peak"],
                "train/mem_reserved_MB": mem_now["reserved"],
                "train/tokens_cum": token_count,
            })

    dt = time.time() - t0
    mem = get_mem_mb()
    print("="*80)
    print("BASELINE (Llama-3 LoRA)")
    print(f"model={args.model} steps={total_steps} batch={args.batch_size}")
    print(f"tokens={token_count} throughput={token_count/max(dt,1e-6):.1f} toks/s")
    print(f"CUDA alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")
    print("="*80)

    if args.save_dir is not None:
        print(f"Saving model and tokenizer to {args.save_dir} ...")
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"Checkpoint saved at {args.save_dir}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
