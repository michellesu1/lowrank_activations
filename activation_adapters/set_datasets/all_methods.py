import argparse
import random
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    LlamaForSequenceClassification,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, LARSConfig, IA3Config, get_peft_model, TaskType

disable_caching()


# run experiments with lowrank_activations/text.sh and lowrank_activations/vision.sh



# ---------------- GLOBAL DEFAULTS ----------------
MODEL_ID_LLM = "meta-llama/Llama-3.2-1B"
MODEL_ID_VIT = "google/vit-base-patch16-224-in21k"

BATCH_SIZE = 8
STEPS = 3000
LR = 5e-4
WARMUP_STEPS = 300
MAX_LEN = 256
ACCUM_STEPS = 4

LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LARS_RANK = 16

DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EVAL_EVERY = 50
DEFAULT_EPOCHS = 10_000 #manual stops for small tests
DEFAULT_PATIENCE = 10 #increase
DEFAULT_SEED = 42

# ---------------- stuff ----------------
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cuda_mem_mb():
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "peak": 0.0, "reserved": 0.0}
    return {
        "alloc": torch.cuda.memory_allocated() / 1e6,
        "peak": torch.cuda.max_memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
    }

def pick_splits(ds_dict):
    if "train" in ds_dict and ("validation" in ds_dict or "test" in ds_dict):
        val = "validation" if "validation" in ds_dict else "test"
        return "train", val
    split = ds_dict["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
    ds_dict["train"] = split["train"]
    ds_dict["validation"] = split["test"]
    return "train", "validation"

# ---------------- TEXT DATASETS ----------------
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

def build_text_dataset(name, tokenizer, max_len, winogrande_config="winogrande_xl", prompt_style="pair"):
    if name == "boolq":
        ds = build_boolq_dataset(tokenizer, max_len)
        return ds, "train", "validation", "boolq_acc"

    if name == "winogrande":
        ds = build_winogrande_dataset(tokenizer, max_len, config_name=winogrande_config, prompt_style=prompt_style)
        val_split = "validation" if "validation" in ds else "test"
        return ds, "train", val_split, "winogrande_acc"

    raise ValueError(f"Unknown text dataset: {name}")

def text_collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = [x["input_ids"] + [pad_id] * (max_len - len(x["input_ids"])) for x in batch]
    attention_mask = [x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch]
    labels = [x["labels"] for x in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# ---------------- VISION DATASETS ----------------
def build_vision_dataset(name, paintings_label="style"):
    if name == "pets":
        ds = load_dataset("timm/oxford-iiit-pet")
        train_split, val_split = pick_splits(ds)
        image_col = "image"
        label_col = "label"
        metric = "pets_acc"
    elif name == "paintings":
        ds = load_dataset("huggan/wikiart")
        train_split, val_split = pick_splits(ds)
        image_col = "image"
        label_col = paintings_label
        metric = f"wikiart_{paintings_label}_acc"
    else:
        raise ValueError(f"Unknown vision dataset: {name}")

    feat = ds[train_split].features.get(label_col, None)
    if isinstance(feat, ClassLabel):
        id2label = {i: n for i, n in enumerate(feat.names)}
        label2id = {n: i for i, n in id2label.items()}
    else:
        uniq = sorted(set(ds[train_split][label_col]))
        id2label = {i: str(v) for i, v in enumerate(uniq)}
        label2id = {str(v): i for i, v in enumerate(uniq)}

    return ds, train_split, val_split, image_col, label_col, label2id, id2label, metric

def vision_collate_fn(batch, processor, image_col, label_col, label2id):
    images = []
    labels = []
    for x in batch:
        img = x[image_col]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        images.append(img)

        v = x[label_col]
        if isinstance(v, int):
            labels.append(label2id.get(str(v), v))
        else:
            labels.append(label2id[str(v)])

    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs

# ---------------- EVAL ----------------
@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        if mode == "text":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values=pixel_values).logits

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / max(1, total)

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--mode", choices=["text", "vision"], required=True)
    ap.add_argument("--method", choices=["lora", "lars", "ia3"], required=True)

    # logging/model
    ap.add_argument("--project", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # global training defaults (apply to ALL modes unless overridden)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--steps", type=int, default=STEPS, help="Max optimizer steps (updates).")
    ap.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument("--accum_steps", type=int, default=ACCUM_STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    ap.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    # text args
    ap.add_argument("--text_dataset", choices=["boolq", "winogrande"], default="winogrande")
    ap.add_argument("--winogrande_config", default="winogrande_xl")
    ap.add_argument("--prompt_style", choices=["pair", "filled"], default="pair")

    # vision args
    ap.add_argument("--vision_dataset", choices=["paintings", "pets"], default="paintings")
    ap.add_argument("--paintings_label", choices=["style", "artist", "genre"], default="style")

    # PEFT hparams
    ap.add_argument("--lora_rank", type=int, default=LORA_RANK)
    ap.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    ap.add_argument("--lars_rank", type=int, default=LARS_RANK)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # defaults for model/project
    if args.mode == "text":
        if args.model is None:
            args.model = MODEL_ID_LLM
        if args.project is None:
            args.project = f"text_{args.method}_{args.text_dataset}_peft"
    else:
        if args.model is None:
            args.model = MODEL_ID_VIT
        if args.project is None:
            args.project = f"vision_{args.method}_{args.vision_dataset}_peft"

    wandb.init(project=args.project, config=vars(args))

    # ----- build data + base model -----
    if args.mode == "text":
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        ds, train_split, val_split, metric_name = build_text_dataset(
            args.text_dataset,
            tokenizer,
            MAX_LEN,
            winogrande_config=args.winogrande_config,
            prompt_style=args.prompt_style,
        )

        train_loader = DataLoader(
            ds[train_split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: text_collate_fn(b, tokenizer.pad_token_id),
        )
        val_loader = DataLoader(
            ds[val_split],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: text_collate_fn(b, tokenizer.pad_token_id),
        )

        base_model = LlamaForSequenceClassification.from_pretrained(args.model, num_labels=2)
        base_model.config.pad_token_id = tokenizer.pad_token_id

        task_type = TaskType.SEQ_CLS
        modules_to_save = ["score"] # TODO

    else:
        ds, train_split, val_split, image_col, label_col, label2id, id2label, metric_name = build_vision_dataset(
            args.vision_dataset,
            paintings_label=args.paintings_label,
        )
        processor = AutoImageProcessor.from_pretrained(args.model)

        train_loader = DataLoader(
            ds[train_split],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: vision_collate_fn(b, processor, image_col, label_col, label2id),
        )
        val_loader = DataLoader(
            ds[val_split],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: vision_collate_fn(b, processor, image_col, label_col, label2id),
        )

        base_model = AutoModelForImageClassification.from_pretrained(
            args.model,
            num_labels=len(id2label),
            id2label={int(i): s for i, s in id2label.items()},
            label2id={s: int(i) for s, i in label2id.items()},
            ignore_mismatched_sizes=True,
        )
        task_type = None
        modules_to_save = ["classifier"] # TODO

    # ----- PEFT config -----

    if args.method == "lora":
        # Choose LoRA targets
        if getattr(args, "lora_target", "all-linear") == "all-linear":
            target_modules = "all-linear"
        elif args.lora_target == "qv":
            target_modules = ["query", "value"]
        else:  # qkv
            target_modules = ["query", "key", "value"]

        peft_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=task_type if args.mode == "text" else None,
            # modules_to_save=modules_to_save,
        )

    elif args.method == "lars":
        peft_cfg = LARSConfig(
            rank=args.lars_rank,
            target_modules="all-linear",
            task_type=task_type if args.mode == "text" else None,  # matches old text setup
            fan_in_fan_out=False,
            # modules_to_save=modules_to_save,
        )

    elif args.method == "ia3":
        # IA3 in IreneTenison/peft requires feedforward_modules
        if args.mode == "text":
            feedforward_modules = ["up_proj", "down_proj", "gate_proj"]
            ia3_task_type = TaskType.SEQ_CLS
        else:
            feedforward_modules = ["intermediate.dense", "output.dense"]
            ia3_task_type = None

        peft_cfg = IA3Config(
            target_modules="all-linear",
            feedforward_modules=feedforward_modules,
            fan_in_fan_out=False,  # safer for standard nn.Linear
            task_type=ia3_task_type,
            # modules_to_save=modules_to_save,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")


    # model.encoder.layer 
    if args.mode == "vision" and args.method == "lars":
        if hasattr(base_model, "vit") and hasattr(base_model.vit, "encoder") and hasattr(base_model.vit.encoder, "layer"):
            base_model.encoder = base_model.vit.encoder # !!!!!
        else:
            raise ValueError("expects a ViT-like model with base_model.vit.encoder.layer")


    model = get_peft_model(base_model, peft_cfg).to(device)
    def is_trainable(name_substr):
        return any(name_substr in n and p.requires_grad for n, p in model.named_parameters())

    # print("score trainable?", is_trainable("score"))
    print("any lars params trainable?", any("lars" in n and p.requires_grad for n,p in model.named_parameters()))

    model.train()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    # ----- train loop -----
    step = 0
    best_acc = 0.0
    patience_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if step >= args.steps:
                break

            # forward
            if args.mode == "text":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                out = model(pixel_values=pixel_values, labels=labels)

            loss = out.loss / args.accum_steps
            loss.backward()

            # update every accum_steps microbatches
            if (batch_idx + 1) % args.accum_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            acc = None
            if step > 0 and (step % args.eval_every == 0):
                acc = evaluate(model, val_loader, device, args.mode)

                if acc > best_acc:
                    best_acc = acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print("early stopping")
                    return

            mem = get_cuda_mem_mb()
            log_dict = {
                "loss": float(loss.item() * args.accum_steps),
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

            wandb.log(log_dict, step=step)

            print(
                f"Epoch {epoch} | Step {step:05d} | Loss {loss.item()*args.accum_steps:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | GradNorm {grad_norm:.2f}"
                + (f" | {metric_name} {acc:.4f}" if acc is not None else "")
            )

            step += 1

        if step >= args.steps:
            break

    print("Training complete.")

if __name__ == "__main__":
    main()
