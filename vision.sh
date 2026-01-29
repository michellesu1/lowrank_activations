#!/usr/bin/env bash
set -euo pipefail
cd ~/lowrank_activations/activation_adapters/set_datasets


echo "Starting PEFT vision grid on $(hostname)"
date

METHODS=("ia3")
MODELS=("google/vit-base-patch16-224-in21k")
VISION_DATASETS=("paintings")
PAINTINGS_LABELS=("style")


# ---------------- RUN GRID ----------------
for method in "${METHODS[@]}"; do
  echo "==============================="
  echo "VISION | METHOD: $method"
  echo "==============================="

  for dataset in "${VISION_DATASETS[@]}"; do

    if [[ "$dataset" == "pets" ]]; then
      echo ">>> VISION | pets | $method"
      python unified.py \
        --mode vision \
        --method "$method" \
        --vision_dataset pets \

    elif [[ "$dataset" == "paintings" ]]; then
      for label in "${PAINTINGS_LABELS[@]}"; do
        echo ">>> VISION | paintings ($label) | $method"
        python unified.py \
          --mode vision \
          --method "$method" \
          --vision_dataset paintings \
          --paintings_label "$label" \
          --eval_every 300
      done
    fi

  done
done

echo "Vision grid complete."
date
