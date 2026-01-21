import math
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForSequenceClassification, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import IA3Config, get_peft_model, TaskType, prepare_model_for_kbit_training

# ---------------- CONFIG ----------------
MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "llama_boolq_peft_quantized"

BATCH_SIZE = 8
STEPS = 3000
LR = 5e-4
WARMUP_STEPS = 300
MAX_LEN = 256
ACCUM_STEPS = 4  # gradient accumulation

# LORA_RANK = 8
# LORA_ALPHA = 32
# LORA_DROPOUT = 0.05

# ---------------- DATASET ----------------
def build_boolq_dataset(tokenizer, max_len):
    ds = load_dataset("boolq")

    def preprocess(ex):
        text = f"Question: {ex['question']}\nPassage: {ex['passage']}"
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(ex["answer"]),  # 0/1 for sequence classification
        }

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)
    return ds

def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id]*(max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0]*(max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ---------------- EVALUATION ----------------
@torch.no_grad()
def evaluate(model, dev_loader, device):
    model.eval()
    correct = total = 0

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total

# ---------------- MAIN ----------------
def main():
    wandb.init(project=PROJECT_NAME, group="ia3_4bit")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    boolq_ds = build_boolq_dataset(tokenizer, MAX_LEN)
    train_loader = DataLoader(
        boolq_ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        boolq_ds["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # 8-bit quantization config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,             
    #     llm_int8_enable_fp32_cpu_offload=False,
    # )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Quantized base model
    base_model = LlamaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        quantization_config=bnb_config,    
        device_map="auto",                 
        torch_dtype="auto",
    )

    # IA³ adapter config
    ia3_config = IA3Config(
        task_type=TaskType.SEQ_CLS,   # sequence classification
        target_modules="all-linear",  # which modules to scale
        fan_in_fan_out=True,              # use fan-in scaling, optional
    )

    # wrap model with IA³
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, ia3_config)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=STEPS,
    )

    # Training loop
    step = 0
    optimizer.zero_grad()
    for epoch in range(1000):
        for batch_idx, batch in enumerate(train_loader):
            if step >= STEPS:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / ACCUM_STEPS
                loss.backward()

            # inside the accumulation step
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                mem_alloc = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_peak  = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                mem_reserved = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0

                # evaluate only every 50 steps
                acc = None
                if step % 50 == 0 and step > 0:
                    acc = evaluate(model, dev_loader, device)

                log_dict = {
                    "loss": float(loss.item() * ACCUM_STEPS),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "mem_allocated_MB": mem_alloc,
                    "mem_peak_MB": mem_peak,
                    "mem_reserved_MB": mem_reserved,
                    "step": step,
                }
                if acc is not None:
                    log_dict["boolq_acc"] = acc

                wandb.log(log_dict)
                print(f"Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} | LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f} | MemAlloc {mem_alloc:.1f}MB" + (f" | Acc {acc:.4f}" if acc is not None else ""))

                step += 1

        if step >= STEPS:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()
