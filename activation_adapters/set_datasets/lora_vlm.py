#!/usr/bin/env python3
"""
ViT on:
  - Paintings: huggan/wikiart (default label is style, also has artist, genre)
  - Pets: timm/oxford-iiit-pet

example:
python activation_adapters/set_datasets/lora_vlm.py --dataset pets --project vit_pets_single_strong   --batch_size 64 --accum_steps 2 --eval_every 50   --lr 1e-4 --warmup_steps 400 --weight_decay 0.01   --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --epochs 120 --steps 4000
"""

#add early stop

import argparse
import math
import os
import random
from typing import Dict, List, Tuple

import torch
import wandb
from datasets import ClassLabel, load_dataset, disable_caching
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
disable_caching()

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_MODEL_ID = "google/vit-base-patch16-224-in21k"
DEFAULT_PROJECT = "vit_paintings_pets_lora"

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_STEPS = 10_000  # max, use epoch mainly
DEFAULT_EVAL_EVERY = 100

DEFAULT_LR = 1e-4
DEFAULT_WARMUP_STEPS = 100
DEFAULT_ACCUM_STEPS = 4
DEFAULT_WEIGHT_DECAY = 0.01

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_splits(ds_dict, prefer_val=True) -> Tuple[str, str]:
    """Return (train_split, val_split) if present; otherwise create one from train."""
    keys = set(ds_dict.keys())
    if "train" in keys and ("validation" in keys or "test" in keys):
        val = "validation" if ("validation" in keys and prefer_val) else ("test" if "test" in keys else "validation")
        return "train", val
    if "train" in keys:

        split = ds_dict["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
        ds_dict["train"] = split["train"]
        ds_dict["validation"] = split["test"]
        return "train", "validation"

    any_split = list(ds_dict.keys())[0]
    split = ds_dict[any_split].train_test_split(test_size=0.1, seed=42, shuffle=True)
    ds_dict["train"] = split["train"]
    ds_dict["validation"] = split["test"]
    return "train", "validation"


def build_pets_dataset() -> Tuple[dict, str, str, str, str]:
    """
    Pets dataset: timm/oxford-iiit-pet
    Expect columns: image, label
    """
    ds = load_dataset("timm/oxford-iiit-pet")
    train_split, val_split = pick_splits(ds, prefer_val=False) 
    image_col = "image"
    label_col = "label"
    metric_name = "pets_acc"
    return ds, train_split, val_split, image_col, label_col, metric_name


def build_paintings_dataset(label_kind: str) -> Tuple[dict, str, str, str, str, str]:
    """
    Paintings dataset: huggan/wikiart
    Columns include: image, artist, genre, style
    label_kind: one of {"style","artist","genre"}
    """
    assert label_kind in {"style", "artist", "genre"}, "paintings_label must be style|artist|genre"
    ds = load_dataset("huggan/wikiart")
    train_split, val_split = pick_splits(ds, prefer_val=True)
    image_col = "image"
    label_col = label_kind
    metric_name = f"wikiart_{label_kind}_acc"
    return ds, train_split, val_split, image_col, label_col, metric_name


def build_label_mapping(ds_dict, train_split: str, label_col: str) -> Tuple[Dict, Dict]:
    """
    Returns (label2id, id2label). Handles:
      - ClassLabel feature
      - int labels
      - string labels
    """
    feat = ds_dict[train_split].features.get(label_col, None)
    if isinstance(feat, ClassLabel):
        id2label = {i: name for i, name in enumerate(feat.names)}
        label2id = {name: i for i, name in id2label.items()}
        return label2id, id2label

    # to avoid scanning everything, sample a chunk then fall back to full unique if needed.
    labels = ds_dict[train_split][label_col]
    # labels can be list of ints/strings
    uniq = sorted(set(labels))

    id2label = {i: str(v) for i, v in enumerate(uniq)}
    label2id = {str(v): i for i, v in enumerate(uniq)}
    return label2id, id2label


def encode_label(x, label_col: str, label2id: Dict) -> int:
    v = x[label_col]
    if isinstance(v, int):

        return label2id.get(str(v), v)
    return label2id[str(v)]


def collate_fn(batch, image_processor, image_col: str, label_col: str, label2id: dict):
    images = []
    labels = []
    for x in batch:
        img = x[image_col]
        #some datasets have RGBA / grayscale so force 3channel RGB
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        images.append(img)
        labels.append(encode_label(x, label_col, label2id))

    proc = image_processor(images=images, return_tensors="pt")
    proc["labels"] = torch.tensor(labels, dtype=torch.long)
    return proc



@torch.no_grad()
def evaluate(model, dev_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in dev_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        logits = model(pixel_values=pixel_values).logits
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


def main():
    ap = argparse.ArgumentParser()

    # dataset + task
    ap.add_argument("--dataset", choices=["paintings", "pets"], default="paintings")
    ap.add_argument("--paintings_label", choices=["style", "artist", "genre"], default="style")

    # model + logging
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--seed", type=int, default=42)

    # training
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Max optimizer steps (updates).")
    ap.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)

    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--accum_steps", type=int, default=DEFAULT_ACCUM_STEPS)
    ap.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    # LoRA
    ap.add_argument("--lora_rank", type=int, default=DEFAULT_LORA_RANK)
    ap.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument(
        "--lora_target",
        choices=["all-linear", "qv", "qkv"],
        default="all-linear",
        help="Which modules to LoRA. all-linear is simplest; qv/qkv are common for ViT attention.",
    )

    args = ap.parse_args()
    seed_everything(args.seed)

    wandb.init(project=args.project, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ----- dataset -----
    if args.dataset == "pets":
        ds, train_split, val_split, image_col, label_col, metric_name = build_pets_dataset()
    else:
        ds, train_split, val_split, image_col, label_col, metric_name = build_paintings_dataset(args.paintings_label)

    # label mapping
    label2id, id2label = build_label_mapping(ds, train_split, label_col)
    num_labels = len(id2label)
    print(f"Using dataset={args.dataset} label={label_col} num_labels={num_labels}")

    # image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model)

    # loaders
    train_loader = DataLoader(
        ds[train_split],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(b, image_processor, image_col, label_col, label2id),
    )
    dev_loader = DataLoader(
        ds[val_split],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_fn(b, image_processor, image_col, label_col, label2id),
    )

    # ----- model -----
    base_model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={int(i): s for i, s in id2label.items()},
        label2id={s: int(i) for s, i in label2id.items()},
        ignore_mismatched_sizes=True,
    )

    # Choose LoRA targets
    if args.lora_target == "all-linear":
        target_modules = "all-linear"
    elif args.lora_target == "qv":
        target_modules = ["query", "value"]
    else:  # qkv
        target_modules = ["query", "key", "value"]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    step = 0  # optimizer updates
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / args.accum_steps
            loss.backward()

            # update every accum_steps mini-batches
            if (batch_idx + 1) % args.accum_steps != 0:
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

            wandb.log(log_dict, step = step)
            print(
                f"Epoch {epoch} | Step {step:05d} | Loss {loss.item()*args.accum_steps:.4f} | "
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
