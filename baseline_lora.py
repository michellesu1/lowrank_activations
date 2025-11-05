# baseline_lora.py
# pip install torch transformers peft accelerate

import argparse, time
import torch
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

def step_peak_mb():
    if not torch.cuda.is_available(): return 0.0
    p = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    return p

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
    target_modules = target_modules or ["c_attn", "c_proj"]  # GPT-2 style
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_dir", default=None, help="Directory to save final model checkpoint")  # NEW
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                     dropout=args.lora_dropout, target_modules=["c_attn","c_proj"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    texts = make_toy_texts(10000)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak()

    t0 = time.time(); token_count = 0
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
        if step % 10 == 0:
            mem = get_mem_mb()
            print(f"[baseline] step {step:03d} | loss {loss.item():.3f} | "
                  f"alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")

    dt = time.time() - t0
    mem = get_mem_mb()
    print("="*80)
    print("BASELINE (LoRA only)")
    print(f"model={args.model} steps={args.steps} batch={args.batch_size}")
    print(f"tokens={token_count} throughput={token_count/max(dt,1e-6):.1f} toks/s")
    print(f"CUDA alloc={mem['alloc']:.1f}MB peak={mem['peak']:.1f}MB reserved={mem['reserved']:.1f}MB")
    print("="*80)

    # --- SAVE CHECKPOINT IF SPECIFIED ---
    if args.save_dir is not None:
        print(f"Saving model and tokenizer to {args.save_dir} ...")
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"Checkpoint saved at {args.save_dir}")

if __name__ == "__main__":
    main()