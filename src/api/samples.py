import os
from pathlib import Path
from typing import List

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_SAMPLES_DIR = DEFAULT_STATIC_DIR / "samples"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def get_samples_dir() -> Path:
    custom = os.getenv("FOOD_SAMPLES_DIR")
    return Path(custom) if custom else DEFAULT_SAMPLES_DIR


def list_sample_paths(limit: int = 10) -> List[Path]:
    root = get_samples_dir()
    if not root.exists():
        return []
    files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in VALID_EXTS]
    return files[:limit]
