import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import random_split, DataLoader

from transformers import AutoModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer
from alpaca_dataset import build_alpaca_dataset, get_alpaca_tokenizer


PROJECT_NAME = "baseline_llama32_1b_like_sdpa_bf16_lora_pt"
ENTITY_NAME = "mchllsu1-massachusetts-institute-of-technology"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

print("W&B URL:", wandb.run.url)


MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
VAL_FRAC = 0.05

BASE_LR = 3e-5
TOTAL_STEPS = 3000
WARMUP_STEPS = 300
VAL_EVERY = 100
ACCUM_STEPS = 4


# ---------------- LoRA linear ----------------

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.05, bias=False):
        super().__init__()
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0

        # base weight (FROZEN)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))          # trainable
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))         # trainable
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

        # init base + LoRA
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if r > 0:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # base linear
        out = F.linear(x, self.weight, self.bias)
        # LoRA branch
        if self.r > 0:
            lora = F.linear(self.lora_dropout(x), self.lora_A)   # (.., r)
            lora = F.linear(lora, self.lora_B)                   # (.., out)
            out = out + self.scaling * lora
        return out


# ---------------- RMSNorm ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


# ---------------- RoPE ----------------

def apply_rope(x, position_ids, rope_theta):
    """
    x: (B, H, S, hd)
    position_ids: (B, S)
    """
    b, h, s, hd = x.shape
    device, dtype = x.device, x.dtype

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=dtype) / hd))
    freqs = torch.einsum("bs,d->bsd", position_ids.to(dtype), inv_freq)
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(1)   # (B,1,S,hd/2)
    cos = cos.unsqueeze(1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x2 * cos + x1 * sin
    return x_rope


# ---------------- GQA Attention (SDPA + LoRA) ----------------

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, rope_theta, attn_dropout=0.0,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.rope_theta = rope_theta

        self.q_proj = LoRALinear(dim, n_heads * self.head_dim,
                                 r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.k_proj = LoRALinear(dim, n_kv_heads * self.head_dim,
                                 r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.v_proj = LoRALinear(dim, n_kv_heads * self.head_dim,
                                 r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.out_proj = LoRALinear(dim, dim,
                                   r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, x, position_ids, attention_mask=None):
        B, S, D = x.shape
        H, H_kv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H_kv, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H_kv, hd).transpose(1, 2)

        q = apply_rope(q, position_ids, self.rope_theta)
        k = apply_rope(k, position_ids, self.rope_theta)

        if H_kv != H:
            repeat = H // H_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        q = q.reshape(B * H, S, hd)
        k = k.reshape(B * H, S, hd)
        v = v.reshape(B * H, S, hd)

        causal = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        if attention_mask is not None:
            pad = (attention_mask == 0)
            pad_4d = pad[:, None, None, :].expand(B, 1, S, S)
            causal_4d = causal.view(1, 1, S, S)
            full_mask = pad_4d | causal_4d
            attn_mask = full_mask.expand(B, H, S, S).reshape(B * H, S, S)
        else:
            attn_mask = causal.view(1, S, S).expand(B * H, S, S)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.reshape(B, H, S, hd).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn_output)


# ---------------- SwiGLU MLP + LoRA ----------------

class LlamaMLP(nn.Module):
    def __init__(self, d_model, ffn_dim, use_bias=False,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.gate_proj = LoRALinear(d_model, ffn_dim,
                                    r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=use_bias)
        self.up_proj   = LoRALinear(d_model, ffn_dim,
                                    r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=use_bias)
        self.down_proj = LoRALinear(ffn_dim, d_model,
                                    r=lora_r, alpha=lora_alpha, dropout=lora_dropout, bias=use_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------- Transformer Block ----------------

class LlamaBlock(nn.Module):
    def __init__(self, d_model, ffn_dim, n_heads, n_kv_heads, rope_theta,
                 attn_dropout=0.0, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(
            d_model, n_heads, n_kv_heads, rope_theta,
            attn_dropout=attn_dropout,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = LlamaMLP(
            d_model, ffn_dim, use_bias=False,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )

    def forward(self, x, position_ids, attention_mask=None):
        x = x + self.attn(self.norm1(x), position_ids, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------- Llama‑3.2‑like 1B ----------------

class Llama3_1B(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_dim,
        rope_theta,
        max_seq_len=256,
        attn_dropout=0.0,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LlamaBlock(
                d_model, ffn_dim, n_heads, n_kv_heads, rope_theta,
                attn_dropout=attn_dropout,
                lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        for layer in self.layers:
            x = layer(x, pos, attention_mask)

        x = self.norm(x)
        return self.lm_head(x)


# ---------------- Validation helper ----------------

def evaluate(model, dev_loader, ce_loss, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
                B, S, V = logits.shape
                loss = ce_loss(
                    logits.reshape(B * S, V),
                    labels.reshape(B * S),
                )

            valid = (labels != ce_loss.ignore_index).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

    model.train()
    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl


# ---------------- TRAINING & EMBEDDING INIT ----------------

def main():
    wandb.init(project=PROJECT_NAME, entity = ENTITY_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_cfg = LlamaConfig.from_pretrained(MODEL_ID)
    d_model    = hf_cfg.hidden_size
    n_layers   = hf_cfg.num_hidden_layers
    n_heads    = hf_cfg.num_attention_heads
    n_kv_heads = hf_cfg.num_key_value_heads
    ffn_dim    = hf_cfg.intermediate_size
    rope_theta = hf_cfg.rope_theta

    # Llama tokenizer + Alpaca dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = build_alpaca_dataset(
        tokenizer,
        max_len=MAX_LEN,
        mask_prompt=True,
    )

    n_total = len(full_dataset)
    n_dev = int(VAL_FRAC * n_total)
    n_train = n_total - n_dev
    train_ds, dev_ds = random_split(
        full_dataset,
        [n_train, n_dev],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    batch0 = next(iter(train_loader))
    print("labels[0][:40]:", batch0["labels"][0][:40])
    print("unique labels:", torch.unique(batch0["labels"]))
    print("ignore_index in CE:", -100)

    hf_lm = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="cpu",
    )

    with torch.no_grad():
        pretrained_emb = hf_lm.model.embed_tokens.weight  # (vocab_size, d_model)

    vocab_size = pretrained_emb.shape[0]
    del hf_lm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model = Llama3_1B(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        rope_theta=rope_theta,
        max_seq_len=MAX_LEN,
        attn_dropout=0.0,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ).to(device).to(torch.bfloat16)

    with torch.no_grad():
        assert pretrained_emb.shape == model.token_emb.weight.shape
        model.token_emb.weight.copy_(pretrained_emb)
        model.lm_head.weight.data.copy_(model.token_emb.weight.data)

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    base_lr = BASE_LR
    total_steps = TOTAL_STEPS
    warmup_steps = WARMUP_STEPS

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=0.01,
    )
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    step = 0
    optimizer.zero_grad()

    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
                B, S, V = logits.shape
                loss = ce_loss(
                    logits.reshape(B * S, V),
                    labels.reshape(B * S),
                )

            loss = loss / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                mem_alloc = mem_peak = mem_reserved = 0.0
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / 1e6
                    mem_peak = torch.cuda.max_memory_allocated() / 1e6
                    mem_reserved = torch.cuda.memory_reserved() / 1e6

                true_loss = float(loss.item() * ACCUM_STEPS)
                wandb.log({
                    "loss": true_loss,
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "mem_allocated_MB": mem_alloc,
                    "mem_peak_MB": mem_peak,
                    "mem_reserved_MB": mem_reserved,
                    "step": step,
                })

                if step % VAL_EVERY == 0 and step > 0:
                    val_loss, val_ppl = evaluate(model, dev_loader, ce_loss, device)
                    wandb.log({
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                        "step": step,
                    })
                    print(f"[VAL] step={step:04d} | val_loss={val_loss:.6f} | val_ppl={val_ppl:.2f}")

                print(
                    f"step={step:04d} | loss={true_loss:.6f} | lr={lr:.2e} | "
                    f"grad_norm={float(grad_norm):.2f} | "
                    f"mem_alloc={mem_alloc:.1f}MB mem_peak={mem_peak:.1f}MB "
                    f"mem_reserved={mem_reserved:.1f}MB"
                )

                step += 1

        # flush leftover grads at end of epoch
        if (batch_idx + 1) % ACCUM_STEPS != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1


        if step >= total_steps:
            break

    print("DONE. Check W&B for train and val curves for baseline Llama3_1B + LoRA.")
    wandb.finish()

if __name__ == "__main__":
    main()
