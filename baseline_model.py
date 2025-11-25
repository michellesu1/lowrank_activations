import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer

wandb.init(project="baseline_model")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.weight * x / (norm + self.eps)

def apply_rope(x, seq_dim=1, rope_theta=10000):
    b, s, nh, hd = x.shape
    device, dtype = x.device, x.dtype
    pos = torch.arange(s, device=device, dtype=dtype)
    inv = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    freqs = torch.outer(pos, inv)
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x2 * cos + x1 * sin
    return x_rope

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, S, D = x.shape
        H, H_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)  # (B, H, S, hd)
        k = self.k_proj(x).view(B, S, H_kv, hd).transpose(1, 2)  # (B, H_kv, S, hd)
        v = self.v_proj(x).view(B, S, H_kv, hd).transpose(1, 2)  # (B, H_kv, S, hd)
        q, k = apply_rope(q), apply_rope(k)
        if H_kv != H:
            k = k.repeat_interleave(H // H_kv, dim=1)
            v = v.repeat_interleave(H // H_kv, dim=1)
        score = torch.matmul(q, k.transpose(-1, -2)) / hd**0.5  # (B, H, S, S)
        mask = torch.triu(torch.ones(S, S, device=x.device), 1).bool()  # [S, S]
        score = score.masked_fill(mask[None, None], float('-inf')) 
        attn = torch.softmax(score, dim=-1)
        z = (attn @ v).transpose(1, 2).reshape(B, S, -1)  # (B, S, D)
        return self.out_proj(z)

class LlamaBlock(nn.Module):
    def __init__(self, d_model, ffn_dim, n_heads, n_kv_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Llama3_1B(nn.Module):
    def __init__(self, vocab_size=32000, n_layers=16, d_model=1024,
                 n_heads=16, n_kv_heads=4, ffn_dim=4096, max_seq_len=2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LlamaBlock(d_model, ffn_dim, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# ------ TRAINING AND LOGGING CODE ------

def main():
    # Use a real tokenizer; adjust model vocab if needed
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size
    seq_len = 24
    batch = 4  # Number of sentences

    # Example real English mini-corpus
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers power modern language models.",
        "Low-rank approximations save GPU memory.",
        "Learning rate schedules help neural nets converge."
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Llama3_1B(
        vocab_size=vocab_size, d_model=1024,
        n_heads=16, n_kv_heads=4, ffn_dim=4096, n_layers=16
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    steps = 40

    for step in range(steps):
        toks = tokenizer(
            sentences * ((batch + len(sentences) - 1) // len(sentences)),
            padding="max_length", truncation=True, max_length=seq_len + 1, return_tensors="pt"
        )
        all_ids = toks["input_ids"][:batch].to(device)
        input_ids = all_ids[:, :-1]
        targets = all_ids[:, 1:]

        logits = model(input_ids)
        logits = logits.contiguous().view(-1, vocab_size)
        targets = targets.contiguous().view(-1)
        loss = ce_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        lr = optimizer.param_groups[0]['lr']
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        optimizer.step()
        scheduler.step()

        # --- Memory logging ---
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

        print(f"step={step:02d} | loss={loss.item():.4f} | lr={lr:.2e} | grad_norm={float(grad_norm):.2f} | "
              f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB mem_reserved={mem_reserved:.1f}MB")

    print("DONE. Check W&B dashboard for full metric and memory curves.")

if __name__ == "__main__":
    main()