import argparse
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, LlamaForSequenceClassification, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Re-show map/filter bars (turn off HF datasets caching)
disable_caching()

#python lora_boolq_winograde.py   --dataset winogrande   --winogrande_config winogrande_xl   --prompt_style pair   --epochs 3 


# ---------------- DEFAULT CONFIG ----------------
MODEL_ID = "meta-llama/Llama-3.2-1B"
PROJECT_NAME = "llama_winogrande_peft"

BATCH_SIZE = 16
STEPS = 10000  # max optimizer steps (updates), not batches
EPOCHS = 3

LR = 1e-4
WARMUP_STEPS = 100
MAX_LEN = 256

ACCUM_STEPS = 4

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ---------------- DATASETS ----------------
def build_boolq_dataset(tokenizer, max_len):
    ds = load_dataset("boolq")

    def preprocess(ex):
        text = f"Question: {ex['question']}\nPassage: {ex['passage']}"
        enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(ex["answer"]),  # 0/1
        }

    return ds.map(preprocess, remove_columns=ds["train"].column_names)


def build_winogrande_dataset(tokenizer, max_len, config_name="winogrande_xl", prompt_style="pair"):
    """
    2-class classification:
      - prompt_style="pair": label is 0 (A) or 1 (B)
      - prompt_style="filled": ask if A is correct; label is 1 yes / 0 no
    """
    ds = load_dataset("winogrande", config_name)

    # Some examples have empty answer -> drop them
    ds = ds.filter(lambda ex: ex["answer"] in ["1", "2"])

    def preprocess(ex):
        sent = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        gold = int(ex["answer"]) - 1  # "1"/"2" -> 0/1

        if prompt_style == "pair":
            text = (
                "Choose the option that best fills the blank.\n"
                f"Sentence: {sent}\n"
                f"Option A: {opt1}\n"
                f"Option B: {opt2}\n"
                "Answer with A or B."
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": gold}

        if prompt_style == "filled":
            completed_a = sent.replace("_", opt1)
            is_a_correct = 1 if gold == 0 else 0
            text = (
                "Is the completed sentence correct?\n"
                f"Completed: {completed_a}\n"
                "Answer: yes or no."
            )
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": is_a_correct}

        raise ValueError(f"Unknown prompt_style: {prompt_style}. Use 'pair' or 'filled'.")

    return ds.map(preprocess, remove_columns=ds["train"].column_names)


def build_dataset(name, tokenizer, max_len, winogrande_config="winogrande_xl", prompt_style="pair"):
    if name == "boolq":
        ds = build_boolq_dataset(tokenizer, max_len)
        return ds, "train", "validation", "boolq_acc"

    if name == "winogrande":
        ds = build_winogrande_dataset(tokenizer, max_len, config_name=winogrande_config, prompt_style=prompt_style)
        val_split = "validation" if "validation" in ds else "test"
        return ds, "train", val_split, "winogrande_acc"

    raise ValueError(f"Unknown dataset: {name}")


# ---------------- DATALOADER ----------------
def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_loaders(ds, train_split, val_split, batch_size, pad_id):
    train_loader = DataLoader(
        ds[train_split],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    dev_loader = DataLoader(
        ds[val_split],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    return train_loader, dev_loader


# ---------------- EVAL ----------------
@torch.no_grad()
def evaluate(model, dev_loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / max(1, total)


def get_cuda_mem_mb():
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "peak": 0.0, "reserved": 0.0}
    return {
        "alloc": torch.cuda.memory_allocated() / 1e6,
        "peak": torch.cuda.max_memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
    }


# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["boolq", "winogrande"], default="winogrande")
    ap.add_argument("--winogrande_config", default="winogrande_xl")
    ap.add_argument("--prompt_style", choices=["pair", "filled"], default="pair")
    ap.add_argument("--project", default=PROJECT_NAME)
    ap.add_argument("--model", default=MODEL_ID)

    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps", type=int, default=STEPS, help="Max optimizer steps (updates).")
    ap.add_argument("--eval_every", type=int, default=50)

    args = ap.parse_args()

    wandb.init(project=args.project, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    ds, train_split, val_split, metric_name = build_dataset(
        args.dataset,
        tokenizer,
        MAX_LEN,
        winogrande_config=args.winogrande_config,
        prompt_style=args.prompt_style,
    )

    train_loader, dev_loader = make_loaders(ds, train_split, val_split, args.batch_size, tokenizer.pad_token_id)

    # Model + LoRA
    base_model = LlamaForSequenceClassification.from_pretrained(args.model, num_labels=2)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=args.steps,
    )

    step = 0  # optimizer step count
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / ACCUM_STEPS
            loss.backward()

            # Update every ACCUM_STEPS mini-batches
            if (batch_idx + 1) % ACCUM_STEPS != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            acc = None
            if step > 0 and (step % args.eval_every == 0):
                acc = evaluate(model, dev_loader, device)

            mem = get_cuda_mem_mb()
            log_dict = {
                "loss": float(loss.item() * ACCUM_STEPS),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm),
                "mem_allocated_MB": mem["alloc"],
                "mem_peak_MB": mem["peak"],
                "mem_reserved_MB": mem["reserved"],
                "step": step,
                "epoch": epoch,
            }
            if acc is not None:
                log_dict[metric_name] = acc

            wandb.log(log_dict)
            print(
                f"Epoch {epoch} | Step {step:04d} | Loss {loss.item()*ACCUM_STEPS:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f}"
                + (f" | {metric_name} {acc:.4f}" if acc is not None else "")
            )

            step += 1
            if step >= args.steps:
                break

        if step >= args.steps:
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
