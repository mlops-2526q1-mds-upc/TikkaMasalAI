Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

## Fine-tune ResNet-18 on Food-101

Prereqs:
- Install deps: `make requirements` (ensures `transformers`, `datasets`, `torch`, etc.)
- Data: local imagefolder at `data/raw/food101/{train,val}/CLASS/*.jpg`, or use the built-in HF dataset

Train:
- Local data (preferred if present):
```bash
make train-resnet18
```
This tries `python src/train/finetune_resnet18.py --data_dir data/raw/food101` and falls back to the HF `food101` dataset if the local folder isn’t found.

- Explicit control:
```bash
python src/train/finetune_resnet18.py --data_dir data/raw/food101 --epochs 15 --learning_rate 5e-4 --fp16
```
Outputs are saved to `models/resnet18-food101/` by default.

Evaluate:
```bash
make eval-resnet18
```

Use the fine-tuned model in code:
```python
from src.models.resnet18 import Resnet18

# point model_path to the training output directory
model = Resnet18(preprocessor_path="models/resnet18-food101", model_path="models/resnet18-food101")
```
