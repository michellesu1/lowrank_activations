import torch
import torch.nn as nn
import torch.nn.functional as F

# Force CPU for debugging; change to "cuda" later if needed
device = "cpu"


class TinyLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, n_layers=2, n_heads=4, ffn_dim=256, max_seq_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, attention_mask=None):
        x = self.token_emb(input_ids)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # (B,S) bool
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.lm_head(x)


def main():
    torch.manual_seed(0)
    vocab_size = 1000
    model = TinyLM(vocab_size=vocab_size).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, S = 4, 16
    for step in range(10):
        input_ids = torch.randint(0, vocab_size, (B, S), device=device)
        labels = input_ids.clone()
        attn = torch.ones_like(input_ids)

        logits = model(input_ids, attention_mask=attn)
        if step == 0:
            print("logits dtype:", logits.dtype)

        Bf, Sf, V = logits.shape
        loss = ce_loss(logits.view(Bf * Sf, V), labels.view(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"step {step} loss {loss.item():.8f}")

if __name__ == "__main__":
    main()
