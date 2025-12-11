import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer
from alpaca_dataset import get_alpaca_dataloader

wandb.init(project="baseline_model_llama32_like")

# ---------------- RMSNorm ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # Closer to standard RMSNorm: use mean of squared elements
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm

# ---------------- RoPE ----------------

def apply_rope(x, position_ids, rope_theta=500000):
    """
    x: (B, H, S, hd)
    position_ids: (B, S)
    """
    b, h, s, hd = x.shape
    device, dtype = x.device, x.dtype

    # (hd/2,)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    # (B, S, hd/2)
    freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype), inv_freq)
    sin, cos = freqs.sin(), freqs.cos()
    # (B, 1, S, hd/2)
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x2 * cos + x1 * sin
    return x_rope

class SimpleMHA(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, attn_mask=None):
        B, S, D = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)  # (B,H,S,hd)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)  # (B,H,S,S)

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))

        if attn_mask is not None:
            # attn_mask here is already (B,1,1,S)
            # expand over heads: (B,1,1,S) -> (B,H,S,S) via broadcast on last dim
            mask = attn_mask.to(torch.bool)  # (B,1,1,S)
            scores = scores.masked_fill(~mask, float("-inf"))

        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)  # (B,H,S,hd)
        z = z.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(z)


# ---------------- GQA Attention ----------------

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, position_ids, attn_mask=None):
        """
        x: (B, S, D)
        position_ids: (B, S)
        attn_mask: (B, 1, S, S) or None, 1 = keep, 0 = mask
        """
        B, S, D = x.shape
        H, H_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)      # (B, H, S, hd)
        k = self.k_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)
        v = self.v_proj(x).view(B, S, H_kv, hd).transpose(1, 2)   # (B, H_kv, S, hd)

        # Apply RoPE to q and k
        q = apply_rope(q, position_ids)                           # (B, H, S, hd)
        k = apply_rope(k, position_ids)                           # (B, H_kv, S, hd)

        # Grouped‑query: expand kv heads if needed
        if H_kv != H:
            k = k.repeat_interleave(H // H_kv, dim=1)             # (B, H, S, hd)
            v = v.repeat_interleave(H // H_kv, dim=1)             # (B, H, S, hd)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)  # (B, H, S, S)

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))

        # Optional attention mask (e.g., padding) in {0,1}
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)                                 # (B, H, S, hd)
        z = z.transpose(1, 2).contiguous().view(B, S, D)          # (B, S, D)
        return self.out_proj(z)

# ---------------- SwiGLU MLP ----------------

class LlamaMLP(nn.Module):
    def __init__(self, d_model, ffn_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=use_bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=use_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ---------------- Transformer Block ----------------

# class LlamaBlock(nn.Module):
#     def __init__(self, d_model, ffn_dim, n_heads, n_kv_heads, attn_dropout=0.0):
#         super().__init__()
#         self.norm1 = RMSNorm(d_model)
#         self.attn = SimpleMHA(d_model, n_heads, attn_dropout=attn_dropout)
#         self.norm2 = RMSNorm(d_model)
#         self.ffn = LlamaMLP(d_model, ffn_dim, use_bias=False)

#     def forward(self, x, position_ids, attn_mask=None):
#         # position_ids unused in this simple version
#         x = x + self.attn(self.norm1(x), attn_mask)
#         x = x + self.ffn(self.norm2(x))
#         return x

class LlamaBlock(nn.Module):
    def __init__(self, d_model, ffn_dim, n_heads, n_kv_heads, attn_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads, attn_dropout=attn_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = LlamaMLP(d_model, ffn_dim, use_bias=False)

    def forward(self, x, position_ids, attn_mask=None):
        x = x + self.attn(self.norm1(x), position_ids, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------- Llama‑3.2‑like 1B ----------------

class Llama3_1B(nn.Module):
    def __init__(
        self,
        vocab_size=128256,   # Llama‑3.2 vocab is ~128k[web:33][web:59]
        n_layers=16,
        d_model=2048,        # 1B: 2048 hidden[web:35]
        n_heads=32,          # 32 attention heads[web:35]
        n_kv_heads=8,        # 8 KV heads (GQA)[web:35]
        ffn_dim=8192,        # 8192 intermediate[web:35]
        max_seq_len=8192,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LlamaBlock(d_model, ffn_dim, n_heads, n_kv_heads, attn_dropout=attn_dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optionally tie weights (common in Llama‑style models)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, S)
        attention_mask: (B, S) in {0,1} or None
        """
        B, S = input_ids.shape
        device = input_ids.device

        # (B, S, D)
        x = self.token_emb(input_ids)

        # position_ids: (B, S)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # HF‑style attn_mask: (B, 1, 1, S) or (B, 1, S, S); here we keep (B, 1, 1, S)
        attn_mask_4d = None
        if attention_mask is not None:
            attn_mask_4d = attention_mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, pos, attn_mask_4d)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)
        return logits

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_loader = get_alpaca_dataloader(
        model_name="distilbert-base-uncased",
        max_len=256,
        batch_size=4,
        shuffle=True,
        mask_prompt=True,
    )

    batch = next(iter(train_loader))
    print("labels[0][:40]:", batch["labels"][0][:40])
    print("unique labels:", torch.unique(batch["labels"]))
    print("ignore_index in CE:", -100)

    vocab_size = tokenizer.vocab_size

    model = Llama3_1B(
        vocab_size=vocab_size,
        d_model=2048,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim=8192,
        n_layers=16,
        max_seq_len=256,
    ).to(device)

    base_lr = 3e-5          # a bit lower to be safe on big random model
    total_steps = 10000     # more training
    warmup_steps = 1000

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * float(step + 1) / warmup_steps
        # cosine decay to 10% of base_lr
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # synthetic sanity check (unchanged)
    B_syn, S_syn = 2, 10
    tokens = torch.randint(0, vocab_size, (B_syn, S_syn + 1), device=device)
    input_ids_syn = tokens[:, :-1]
    labels_syn    = tokens[:, 1:]
    attn_syn      = torch.ones_like(input_ids_syn)

    with torch.no_grad():
        logits_syn = model(input_ids_syn, attention_mask=attn_syn)
        Bf, Sf, V = logits_syn.shape
        syn_loss = ce_loss(
            logits_syn.reshape(Bf * Sf, V),
            labels_syn.reshape(Bf * Sf),
        )
        print("Synthetic loss:", syn_loss.item())

    step = 0
    for epoch in range(1000):
        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # update LR from warmup+cosine schedule
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            logits = model(input_ids, attention_mask=attention_mask)
            B, S, V = logits.shape
            loss = ce_loss(
                logits.reshape(B * S, V),
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
                "loss": loss.item(),
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

    print("DONE. Check W&B dashboard for Alpaca training curves.")

if __name__ == "__main__":
    main()