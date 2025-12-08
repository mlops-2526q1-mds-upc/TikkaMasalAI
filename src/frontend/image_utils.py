from __future__ import annotations

import io
from typing import Tuple

from PIL import Image, ImageOps


def prepare_image_for_upload(
    pil_img: Image.Image,
    filename: str,
    max_side: int = 1024,
    target_bytes: int = 950_000,
    min_quality: int = 50,
    start_quality: int = 85,
) -> Tuple[bytes, str, str, Tuple[int, int], bool]:
    """EXIF-correct, resize, and JPEG-compress an image for upload.

    Returns a tuple of (bytes, mime, filename, (w, h), was_modified).
    """
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    rgb = pil_img.convert("RGB")

    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        rgb = rgb.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = w, h

    out_name = filename.rsplit(".", 1)[0] + ".jpg"
    out_mime = "image/jpeg"

    def _save_with_quality(q: int) -> bytes:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        return buf.getvalue()

    quality = start_quality
    upload_bytes = _save_with_quality(quality)
    # First, reduce quality in steps until hitting the target or min quality
    while len(upload_bytes) > target_bytes and quality > min_quality:
        quality = max(min_quality, quality - 5)
        upload_bytes = _save_with_quality(quality)

    # If still too large, progressively downscale dimensions until under target or reaching a floor
    min_side_floor = 640
    while len(upload_bytes) > target_bytes and max(new_w, new_h) > min_side_floor:
        scale = 0.85
        new_w, new_h = max(1, int(round(new_w * scale))), max(1, int(round(new_h * scale)))
        rgb = rgb.resize((new_w, new_h), Image.LANCZOS)
        quality = start_quality  # try better quality at smaller size, then step down again
        upload_bytes = _save_with_quality(quality)
        while len(upload_bytes) > target_bytes and quality > min_quality:
            quality = max(min_quality, quality - 5)
            upload_bytes = _save_with_quality(quality)

    return upload_bytes, out_mime, out_name, (new_w, new_h), True
