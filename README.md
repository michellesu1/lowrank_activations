# lr_activations

## Installation

To install the required packages, run:

`pip install -r requirements.txt`


## Running Experiments

### Baseline Model

`python baseline_lora.py --steps 200 --batch_size 32`

### Modified Model

`python modified_lora.py --steps 200 --batch_size 32 --svd_method svd --svd_rank 32 --svd_collect 30`
