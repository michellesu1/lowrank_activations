#!/usr/bin/env bash
set -euo pipefail

echo "Starting PEFT grid (text + vision) on $(hostname)"
date

# ---------------- PATH ----------------
cd ~/lowrank_activations/activation_adapters/set_datasets

echo "Running from:"
pwd
ls unified.py

METHODS=("lars" "ia3")

# ---------------- TEXT ----------------
for method in "${METHODS[@]}"; do
  echo "==============================="
  echo "TEXT | METHOD: $method"
  echo "==============================="

  # echo ">>> TEXT | boolq | $method"
  # python unified.py \
  #   --mode text \
  #   --method "$method" \
  #   --text_dataset boolq \
  #   --patience 1000000000

  echo ">>> TEXT | winogrande (pair) | $method"
  python unified.py \
    --mode text \
    --method "$method" \
    --text_dataset winogrande \
    --winogrande_config winogrande_xl \
    --prompt_style pair \
    --batch_size 16 \
    --steps 10000 \
    --lr 1e-4 \
    --warmup_steps 100 \
    --patience 10000000000 \
    --eval_every 100

  # echo ">>> TEXT | winogrande (filled) | $method"
  # python unified.py \
  #   --mode text \
  #   --method "$method" \
  #   --text_dataset winogrande \
  #   --winogrande_config winogrande_xl \
  #   --prompt_style filled \
  #   --patience 1000000000
done

echo "Grid complete."
date