"""
Food dataset utilities for Deepchecks and PyTorch.

This module provides:
- FoodDataset: A Dataset that reads images stored in parquet/polars DataFrames
  where the image column may contain bytes, base64 strings, numpy arrays, or dicts
  holding one of those forms. It outputs HWC uint8 numpy arrays compatible with
  deepchecks.vision.VisionData.
- dc_collate: A collate function for DataLoader that returns a dict with
  "images" and "labels" lists as expected by Deepchecks.

Notes
- This implementation intentionally avoids filesystem path loading; it only
  decodes in-memory representations (bytes/base64/arrays). If your data contains
  file paths, load the bytes upstream or extend _to_pil accordingly.
- Images are emitted as HWC uint8 arrays and reshaped to 3 channels (RGB) if
  grayscale, and alpha channels are dropped if present (RGBA -> RGB).
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _build_label_to_idx(class_names: Sequence[str]) -> Dict[str, int]:
    """Build a mapping from class name to index.

    Parameters
    - class_names: ordered sequence of class names; index is the class id

    Returns
    - dict mapping class name -> integer index
    """
    return {name: i for i, name in enumerate(class_names)}


def _to_label_idx(label_val: Any, label_to_idx: Mapping[str, int]) -> int:
    """Normalize an incoming label (int/str/dict) to an integer index.

    Accepted forms:
    - int or np.integer: returned as-is
    - str: either a class name present in label_to_idx or a numeric string
    - dict: tries common keys recursively: 'index', 'id', 'label', 'class_id', 'value'

    Raises KeyError if the value cannot be mapped.
    """
    if isinstance(label_val, (int, np.integer)):
        return int(label_val)
    if isinstance(label_val, str):
        if label_val in label_to_idx:
            return label_to_idx[label_val]
        if label_val.isdigit():
            return int(label_val)
    if isinstance(label_val, dict):
        for k in ("index", "id", "label", "class_id", "value"):
            if k in label_val:
                return _to_label_idx(label_val[k], label_to_idx)
    raise KeyError(f"Cannot map label {label_val!r} to an index; update label mapping handling.")


class FoodDataset(Dataset):
    """Dataset for food classification images stored in tabular data.

    This dataset decodes images stored directly in a DataFrame column into HWC uint8 numpy arrays
    and returns tuples of (image_array, label_index). Intended for use with Deepchecks VisionData
    which expects batch dicts with 'images' and 'labels' keys.

    Parameters
    - df: a DataFrame-like object (tested with polars.DataFrame) with columns for image and label
    - class_names: ordered list/sequence of class names; their indices define the class ids
    - image_col: name of the image column (default: 'image')
    - label_col: name of the label column (default: 'label')
    - transform: optional transform applied to PIL Image; may return PIL or Torch tensor (CHW). The
                 result is converted to HWC uint8 for Deepchecks.
    """

    def __init__(
        self,
        df: Any,
        class_names: Sequence[str],
        image_col: str = "image",
        label_col: str = "label",
        transform: Optional[Any] = None,
    ) -> None:
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.class_names = list(class_names)
        self.label_to_idx = _build_label_to_idx(self.class_names)

        # If polars, keep only needed columns to reduce memory footprint
        try:
            self.df = self.df.select([self.image_col, self.label_col])  # type: ignore[attr-defined]
        except Exception:
            # For non-polars types, ignore
            pass

    def __len__(self) -> int:
        # polars has 'height'; pandas/others support len()
        if hasattr(self.df, "height"):
            return int(getattr(self.df, "height"))
        return len(self.df)

    def _to_pil(self, img_val: Any) -> Image.Image:
        """Decode an image value into a RGB PIL Image.

        Supported types:
        - bytes/bytearray
        - str base64
        - numpy arrays or nested Python lists
        - dict wrappers with keys: bytes/data/image_bytes/content OR b64/base64/image_base64 OR array/ndarray/pixels
        """
        # Raw bytes
        if isinstance(img_val, (bytes, bytearray)):
            return Image.open(BytesIO(img_val)).convert("RGB")

        # Dict-like container
        if isinstance(img_val, dict):
            # bytes under common keys
            for k in ("bytes", "data", "image_bytes", "content"):
                v = img_val.get(k)
                if isinstance(v, (bytes, bytearray)):
                    return Image.open(BytesIO(v)).convert("RGB")
            # base64 strings
            for k in ("b64", "base64", "image_base64"):
                v = img_val.get(k)
                if isinstance(v, str):
                    raw = base64.b64decode(v)
                    return Image.open(BytesIO(raw)).convert("RGB")
            # array-like
            for k in ("array", "ndarray", "pixels"):
                if k in img_val:
                    arr = np.array(img_val[k])
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    if arr.ndim == 2:
                        return Image.fromarray(arr, mode="L").convert("RGB")
                    return Image.fromarray(arr)
            raise TypeError(f"Unsupported image dict structure: keys={list(img_val.keys())}")

        # Numpy/list
        if isinstance(img_val, np.ndarray):
            arr = img_val
        elif isinstance(img_val, list):
            arr = np.array(img_val)
        elif isinstance(img_val, str):
            # Treat plain string as base64 only
            raw = base64.b64decode(img_val)
            return Image.open(BytesIO(raw)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image value type: {type(img_val)}")

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # polars supports .row; fallback to generic indexing
        if hasattr(self.df, "row"):
            row = self.df.row(idx, named=True)  # type: ignore[attr-defined]
            img_val = row[self.image_col]
            label_val = row[self.label_col]
        else:
            # attempt pandas-like access
            try:
                img_val = self.df[self.image_col].iloc[idx]
                label_val = self.df[self.label_col].iloc[idx]
            except Exception:
                # very generic fallback
                rec = self.df[idx]
                img_val = rec[self.image_col]
                label_val = rec[self.label_col]

        pil_img = self._to_pil(img_val)

        # Apply optional transform; normalize to HWC uint8
        if self.transform is not None:
            t_img = self.transform(pil_img)
            if isinstance(t_img, torch.Tensor):
                np_img = t_img.detach().cpu().numpy()
                if np_img.ndim == 3 and np_img.shape[0] in (1, 3):  # CHW -> HWC
                    np_img = np.transpose(np_img, (1, 2, 0))
                # scale floats 0..1 -> 0..255
                if np_img.dtype in (np.float16, np.float32, np.float64):
                    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
                else:
                    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
            else:
                # assume PIL Image
                np_img = np.array(t_img)
        else:
            np_img = np.array(pil_img)

        # Ensure 3 channels HWC
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        if np_img.shape[2] == 4:  # RGBA -> RGB
            np_img = np_img[:, :, :3]

        label_idx = _to_label_idx(label_val, self.label_to_idx)
        # Optionally clamp/validate range
        if label_idx < 0 or label_idx >= len(self.class_names):
            raise ValueError(
                f"Label index {label_idx} out of range [0, {len(self.class_names) - 1}] for value={label_val!r}"
            )
        return np_img, int(label_idx)


def dc_collate(batch: Iterable[Tuple[np.ndarray, int]]) -> Dict[str, List[Any]]:
    """Collate list of (np_image, int_label) tuples into a Deepchecks dict.

    Returns a dict with:
    - images: list of HWC uint8 numpy arrays
    - labels: list of ints
    """
    imgs, labels = zip(*batch)
    return {"images": list(imgs), "labels": [int(label) for label in labels]}
