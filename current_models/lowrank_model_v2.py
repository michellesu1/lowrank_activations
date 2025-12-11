import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer
from alpaca_dataset import get_alpaca_dataloader

wandb.init(project="lowrank_model_llama32_like")

# ---------------- RMSNorm ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.weight * x / (norm + self.eps)

# ---------------- RoPE (MHA-style) ----------------

def apply_rope(x, seq_dim=2, rope_theta=10000):
    """
    x: (B, H, S, hd) with seq_dim = 2
    """
    b, h, s, hd = x.shape
    device, dtype = x.device, x.dtype
    pos = torch.arange(s, device=device, dtype=dtype)           # (S,)
    inv = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    freqs = torch.outer(pos, inv)                               # (S, hd/2)
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(0).unsqueeze(0)                         # (1,1,S,hd/2)
    cos = cos.unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x2 * cos + x1 * sin
    return x_rope

# ---------------- Low-rank embedding (trainable Vk, 2048 -> k) ----------------

class LowRankEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, k, Vk_init):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # 2048-dim embedding
        # Vk_init: (d_model, k) from PCA; make it trainable
        self.Vk = nn.Parameter(Vk_init)                 # (2048, k), trainable

    def forward(self, input_ids):
        x = self.embed(input_ids)      # (B, S, d_model)
        return x @ self.Vk             # (B, S, k)

# ---------------- GQA Attention in low-rank space ----------------

class GQAttention_LR(nn.Module):
    def __init__(self, k, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = k // n_heads

        self.q_proj = nn.Linear(k, k, bias=False)
        self.k_proj = nn.Linear(k, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(k, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(k, k, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, attention_mask=None):
        """
        x: (B, S, K)
        attention_mask: (B, S) or None
        """
        B, S, K = x.shape
        H, H_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)      # (B, H, S, hd)
        k = self.k_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)
        v = self.v_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)

        q = apply_rope(q)
        k = apply_rope(k)

        if H_kv != H:
            k = k.repeat_interleave(H // H_kv, dim=1)
            v = v.repeat_interleave(H // H_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)  # (B,H,S,S)

        # causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))

        # padding mask: attention_mask (B,S) -> (B,1,1,S)
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~pad, float("-inf"))

        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)                                # (B,H,S,hd)
        z = z.transpose(1, 2).contiguous().view(B, S, K)         # (B,S,K)
        return self.out_proj(z)

# ---------------- FFN ----------------

class LlamaBlock_LR(nn.Module):
    def __init__(self, k, ffn_dim, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(k)
        self.attn = GQAttention_LR(k, n_heads, n_kv_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(k)
        self.ffn = nn.Sequential(
            nn.Linear(k, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, k),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------- Low-rank Llama-1B-like model ----------------

class Llama3_1B_LowRank(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=2048,
        k=384,
        Vk_init=None,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim=8192,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = k

        assert Vk_init is not None and Vk_init.shape == (d_model, k)
        self.embed = LowRankEmbedding(vocab_size, d_model, k, Vk_init=Vk_init)

        self.layers = nn.ModuleList([
            LlamaBlock_LR(k, ffn_dim, n_heads, n_kv_heads, attn_dropout=attn_dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(k)
        self.lm_head = nn.Linear(k, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)          # (B,S,k)
        for layer in self.layers:
            x = layer(x, attention_mask)   # (B,S,k)
        x = self.norm(x)
        return self.lm_head(x)             # (B,S,vocab)

# ---------------- TRAINING on Alpaca with semantic PCA Vk ----------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Alpaca dataloader (same interface as your baseline)
    tokenizer, train_loader = get_alpaca_dataloader(
        model_name="distilbert-base-uncased",
        max_len=256,
        batch_size=4,
        shuffle=True,
        mask_prompt=True,
    )

    batch0 = next(iter(train_loader))
    print("labels[0][:40]:", batch0["labels"][0][:40])
    print("unique labels:", torch.unique(batch0["labels"]))
    print("ignore_index in CE:", -100)

    vocab_size = tokenizer.vocab_size

    d_model = 2048
    k = 384
    n_layers = 16
    n_heads = 32
    n_kv_heads = 8
    ffn_dim = 8192

    # ---- Step 1: semantic PCA init for Vk using this model's own 2048-d embeddings ----
    # Build a temporary 2048-d embedding to collect activations
    embed_tmp = nn.Embedding(vocab_size, d_model).to(device)

    acts = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 200:   # limit for speed; increase for better PCA
                break
            ids = batch["input_ids"].to(device)         # (B,S)
            a = embed_tmp(ids).reshape(-1, d_model).cpu()  # (B*S, d_model)
            acts.append(a)
    acts = torch.cat(acts, dim=0)                       # (N, d_model)
    print("PCA source acts shape:", acts.shape)

    # PCA to get top-k basis directions in 2048-d space
    U, S, V = torch.pca_lowrank(acts, q=k, center=True)  # V: (d_model, d_model)[web:258]
    Vk_init = V[:, :k].contiguous()                      # (d_model, k) = (2048, k)

    # ---- Step 2: build low-rank Llama-1B-like model with trainable Vk_init ----
    model = Llama3_1B_LowRank(
        vocab_size=vocab_size,
        d_model=d_model,
        k=k,
        Vk_init=Vk_init,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        attn_dropout=0.0,
    ).to(device)

    # Everything in model.parameters(), including Vk, is trainable
    base_lr = 3e-5
    total_steps = 5000
    warmup_steps = 500

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    step = 0
    for epoch in range(1000):
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(input_ids, attention_mask=attention_mask)  # (B,S,V)
            B, S, Vocab = logits.shape
            loss = ce_loss(
                logits.reshape(B * S, Vocab),
                labels.reshape(B * S),
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mem_alloc = mem_peak = mem_reserved = 0.0
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e6
                mem_peak = torch.cuda.max_memory_allocated() / 1e6
                mem_reserved = torch.cuda.memory_reserved() / 1e6

            wandb.log({
                "loss": float(loss.item()),
                "learning_rate": lr,
                "grad_norm": float(grad_norm),
                "mem_allocated_MB": mem_alloc,
                "mem_peak_MB": mem_peak,
                "mem_reserved_MB": mem_reserved,
                "step": step,
            })

            print(
                f"step={step:04d} | loss={loss.item():.6f} | lr={lr:.2e} | "
                f"grad_norm={float(grad_norm):.2f} | "
                f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB "
                f"mem_reserved={mem_reserved:.1f}MB"
            )

            step += 1
        if step >= total_steps:
            break

    print("DONE. Check W&B dashboard for low-rank Alpaca training curves.")

if __name__ == "__main__":
    main()
