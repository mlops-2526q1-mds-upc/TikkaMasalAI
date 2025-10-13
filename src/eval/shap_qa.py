#!/usr/bin/env python3
"""
Minimal SHAP QA utility for Food-101 models.

This script computes quick SHAP attributions for a small sample of
validation images and saves a few lightweight artifacts for visual QA.

Supported models (by class name): Resnet18, VGG16, PrithivMlFood101.
"""

import argparse
import glob
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image as PILImage
import shap
import torch

from src.labels import LABELS
from src.models.prithiv_ml_food101 import PrithivMlFood101
from src.models.resnet18 import Resnet18
from src.models.vgg16 import VGG16


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_validation_images(
    data_dir: str, max_samples: int, random_seed: int
) -> list[tuple[PILImage.Image, int]]:
    """
    Load up to max_samples validation images as PIL along with labels from parquet files.
    """
    rng = np.random.default_rng(random_seed)
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    if not val_files:
        raise ValueError(f"No parquet files found under {data_dir}")

    samples = []
    for fp in val_files:
        if len(samples) >= max_samples:
            break
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            if len(samples) >= max_samples:
                break
            img_bytes = row["image"]["bytes"]
            label = int(row["label"])
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            samples.append((pil_img, label))

    # simple shuffle for variety
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    samples = [samples[i] for i in indices]
    return samples


def load_model(model_name: str, model_path: str | None = None):
    name = model_name.lower()
    if name in {"resnet18", "resnet-18", "resnet"}:
        if model_path:
            return "resnet18", Resnet18.from_pretrained(model_path)
        return "resnet18", Resnet18()
    if name in {"vgg16", "vgg"}:
        return "vgg16", VGG16()
    if name in {"prithiv", "prithivml", "prithivml_food101", "food101", "food-101"}:
        return "prithiv_ml_food101", PrithivMlFood101()
    raise ValueError(
        "Unsupported model name. Use one of: resnet18, vgg16, prithiv"
    )


def make_predict_fn(model_key: str, model_obj) -> Callable[[Sequence[PILImage.Image]], np.ndarray]:
    """
    Return a callable(images) -> (N, C) numpy float array of class probabilities.
    The callable accepts a sequence of PIL images.
    """
    softmax = torch.nn.Softmax(dim=1)

    if model_key == "vgg16":
        vgg: VGG16 = model_obj

        def predict_vgg(images: Sequence[PILImage.Image]) -> np.ndarray:
            tensors = []
            for img in images:
                tensors.append(vgg.transform(img))
            batch = torch.stack(tensors, dim=0)
            with torch.no_grad():
                logits = vgg.model(batch)
                probs = softmax(logits).cpu().numpy()
            return probs

        return predict_vgg

    if model_key == "resnet18":
        res: Resnet18 = model_obj

        def predict_resnet(images: Sequence[PILImage.Image]) -> np.ndarray:
            inputs = res.image_processor(list(images), return_tensors="pt")
            with torch.no_grad():
                logits = res.model(**inputs).logits
                probs = softmax(logits).cpu().numpy()
            return probs

        return predict_resnet

    if model_key == "prithiv_ml_food101":
        pri: PrithivMlFood101 = model_obj

        def predict_prithiv(images: Sequence[PILImage.Image]) -> np.ndarray:
            inputs = pri.processor(images=list(images), return_tensors="pt")
            with torch.no_grad():
                logits = pri.model(**inputs).logits
                probs = softmax(logits).cpu().numpy()
            return probs

        return predict_prithiv

    raise ValueError(f"Unknown model key {model_key}")


def ensure_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_meta_json(out_dir: str, meta: dict) -> None:
    ensure_dirs(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_shap_qa(
    model_name: str,
    device: str,
    background_k: int,
    eval_n: int,
    seed: int,
    model_path: str | None = None,
) -> None:
    set_seed(seed)

    model_key, model = load_model(model_name, model_path=model_path)
    if device == "cuda" and torch.cuda.is_available():
        # Note: HF models handle device internally; for simplicity we stick to CPU
        pass

    data_dir = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "food101" / "data")

    total_needed = background_k + eval_n
    samples = load_validation_images(data_dir, total_needed, seed)
    if len(samples) < total_needed:
        raise ValueError(
            f"Requested {total_needed} samples but only found {len(samples)} in {data_dir}"
        )

    # Separate background and eval sets
    background_images = [img for img, _ in samples[:background_k]]
    eval_images = [img for img, _ in samples[background_k:background_k + eval_n]]

    # Normalize image sizes to a fixed resolution expected by SHAP and models
    target_size = (224, 224)
    background_images = [img.resize(target_size, resample=PILImage.BILINEAR) for img in background_images]
    eval_images = [img.resize(target_size, resample=PILImage.BILINEAR) for img in eval_images]

    predict_fn = make_predict_fn(model_key, model)

    # SHAP image masker operates on HxWxC uint8 arrays; convert PIL to numpy
    # FIX: Stack into a single array instead of keeping as list
    background_np = np.stack([np.array(img) for img in background_images], axis=0)
    eval_np = np.stack([np.array(img) for img in eval_images], axis=0)

    masker = shap.maskers.Image("inpaint_telea", background_np[0].shape)

    # After creating predict_fn, add this test:
    test_pred = predict_fn([eval_images[0]])
    print(f"Test prediction shape: {test_pred.shape}, sum: {test_pred.sum()}")
    assert test_pred.shape == (1, 101), f"Expected shape (1, 101), got {test_pred.shape}"

    explainer = shap.Explainer(predict_fn, masker, output_names=LABELS)

    shap_values = explainer(eval_np, max_evals=500)

    # Output directories
    base_out = Path(__file__).resolve().parents[2] / "reports"
    figs_dir = base_out / "figures" / "shap" / model_key
    arrays_dir = base_out / "shap" / model_key
    ensure_dirs(str(figs_dir))
    ensure_dirs(str(arrays_dir))

    # Save a couple of plots
    try:
        import matplotlib.pyplot as plt

        # Save per-image overlays for first 4 samples to keep it light
        num_to_plot = min(4, len(eval_np))
        for i in range(num_to_plot):
            plt.figure(figsize=(5, 5))
            shap.image_plot(shap_values[i:i+1], eval_np[i:i+1], show=False)
            plt.tight_layout()
            plt.savefig(figs_dir / f"image_{i}_overlay.png", dpi=150)
            plt.close()
    except Exception as e:
        # Do not fail the run just because of plotting issues
        print(f"Plotting failed: {e}")

    # Save raw attributions for reproducibility (compressed)
    try:
        # Support both a single Explanation and a list of Explanations
        if isinstance(shap_values, list):
            values_arr = np.stack([sv.values for sv in shap_values], axis=0)
            base_values_arr = np.stack([sv.base_values for sv in shap_values], axis=0)
        else:
            sv_any: Any = shap_values
            values_arr = sv_any.values
            base_values_arr = sv_any.base_values

        np.savez_compressed(
            arrays_dir / "shap_values_first_batch.npz",
            values=values_arr,
            base_values=base_values_arr,
        )
    except Exception as e:
        print(f"Saving arrays failed: {e}")

    # Meta
    save_meta_json(
        str(arrays_dir),
        {
            "model": model_key,
            "background_k": background_k,
            "eval_n": eval_n,
            "seed": seed,
            "data_dir": data_dir,
        },
    )

    print(f"Saved SHAP QA artifacts under: {figs_dir} and {arrays_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal SHAP QA for Food-101 models")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--background-k", type=int, default=24)
    parser.add_argument("--eval-n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None, help="Optional path or HF id for model weights (used for Resnet18)")
    args = parser.parse_args()

    run_shap_qa(
        model_name=args.model,
        device=args.device,
        background_k=args.background_k,
        eval_n=args.eval_n,
        seed=args.seed,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()