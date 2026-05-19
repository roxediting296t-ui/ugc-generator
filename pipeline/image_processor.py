"""
pipeline/image_processor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles:
  - Product placement with drop shadow + blend
  - Background blur (portrait mode simulation)
  - Face detection + preprocessing for LivePortrait
  - Background replacement (solid color or custom image)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

log = logging.getLogger("ugc.image")


def merge_images(
    human_image_path: str,
    product_image_path: str,
    job_dir: Path,
    placement: str = "bottom_right",
    product_scale: float = 0.28,
    custom_x: Optional[int] = None,
    custom_y: Optional[int] = None,
    enable_bg_blur: bool = False,
    bg_blur_strength: int = 8,
) -> Path:
    """
    Composite product PNG onto human image with:
    - Drop shadow for depth
    - Edge feathering for natural blend
    - Optional background blur (portrait mode feel)

    ┌──────────────────────────────────────────────────────┐
    │  AI INPAINTING SLOT — Hand Holding Product          │
    │                                                      │
    │  After this composite, call:                        │
    │  ComfyUI: POST http://localhost:8188/prompt         │
    │  A1111:   POST http://localhost:7860/sdapi/v1/img2img│
    │                                                      │
    │  Params: init_image=merged, mask=hand_region,       │
    │  prompt="hand holding product, photorealistic",     │
    │  denoising_strength=0.60                            │
    └──────────────────────────────────────────────────────┘
    """
    log.info("[IMG] Compositing product onto base image...")

    base = Image.open(human_image_path).convert("RGBA")
    product = Image.open(product_image_path).convert("RGBA")

    base_w, base_h = base.size

    # Optional background blur — simulates portrait mode / shallow DoF
    if enable_bg_blur:
        base = _apply_portrait_blur(base, blur_strength=bg_blur_strength)

    # Scale product
    new_prod_w = int(base_w * product_scale)
    ratio = new_prod_w / product.width
    new_prod_h = int(product.height * ratio)
    product = product.resize((new_prod_w, new_prod_h), Image.LANCZOS)
    prod_w, prod_h = product.size
    margin = int(base_w * 0.05)

    placement_map = {
        "bottom_right": (base_w - prod_w - margin, base_h - prod_h - margin),
        "bottom_left":  (margin, base_h - prod_h - margin),
        "top_right":    (base_w - prod_w - margin, margin),
        "top_left":     (margin, margin),
        "center":       (base_w // 2 - prod_w // 2, base_h // 2 - prod_h // 2),
        "hand":         (base_w // 2 - prod_w // 2, int(base_h * 0.62)),
        "custom":       (custom_x or margin, custom_y or margin),
    }
    pos_x, pos_y = placement_map.get(placement, placement_map["bottom_right"])

    # Drop shadow layer
    shadow = _create_drop_shadow(product, offset=(10, 10), blur_radius=15, opacity=110)
    shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_layer.paste(shadow, (pos_x, pos_y), shadow)
    base = Image.alpha_composite(base, shadow_layer)

    # Feather product edges for natural blend
    product = _feather_edges(product, feather_px=4)

    # Paste product
    product_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    product_layer.paste(product, (pos_x, pos_y), product)
    merged = Image.alpha_composite(base, product_layer)

    output_path = job_dir / "merged_base.png"
    merged.convert("RGB").save(str(output_path), "PNG", quality=100)
    log.info(f"[IMG] Merged image saved: {output_path}")
    return output_path


def _apply_portrait_blur(image: Image.Image, blur_strength: int = 8) -> Image.Image:
    """
    Simulates portrait mode — blurs background, keeps center sharp.
    Uses a simple radial gradient mask (no segmentation model needed).
    For production, replace with MediaPipe selfie segmentation.
    """
    w, h = image.size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_strength))

    # Create radial mask — center = sharp, edges = blurred
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = w // 2, h // 3  # face tends to be upper-center
    rx, ry = int(w * 0.35), int(h * 0.45)

    for i in range(min(rx, ry), 0, -1):
        alpha = int(255 * (i / min(rx, ry)) ** 0.5)
        draw.ellipse(
            [cx - i, cy - i, cx + i, cy + i],
            fill=alpha
        )

    return Image.composite(image, blurred, mask)


def _create_drop_shadow(
    img: Image.Image,
    offset: Tuple[int, int] = (8, 8),
    blur_radius: int = 12,
    opacity: int = 110,
) -> Image.Image:
    """Creates a soft drop shadow behind product."""
    alpha = img.split()[-1]
    shadow = Image.new("RGBA", img.size, (0, 0, 0, opacity))
    shadow_masked = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_masked.paste(shadow, mask=alpha)
    shadow_masked = shadow_masked.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ox, oy = offset
    # Crop to stay in bounds
    cx = max(0, min(ox, img.width - 1))
    cy = max(0, min(oy, img.height - 1))
    canvas.paste(shadow_masked, (cx, cy), shadow_masked)
    return canvas


def _feather_edges(img: Image.Image, feather_px: int = 4) -> Image.Image:
    """Softens product edges to avoid hard compositing artifacts."""
    if feather_px <= 0:
        return img
    alpha = img.split()[-1]
    alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=feather_px))
    img.putalpha(alpha_blurred)
    return img


def validate_and_preprocess_face(image_path: Path, job_dir: Path) -> Path:
    """
    Face detection + crop + resize to 512×512 for LivePortrait.

    Uses OpenCV Haar cascade (no GPU needed for detection).
    Raises ValueError immediately if no face found — better than
    wasting 5 minutes of GPU time on bad input.
    """
    log.info("[FACE] Detecting face...")

    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for better detection in varied lighting
    gray = cv2.equalizeHist(gray)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face detected in base image.\n"
            "Requirements:\n"
            "  • Front-facing photo\n"
            "  • Face must be at least 60×60 pixels\n"
            "  • Good lighting, no heavy shadows on face\n"
            "  • AI-generated portraits (Midjourney/SDXL) work well"
        )

    # Use largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    log.info(f"[FACE] Face detected at ({x},{y}) size {w}×{h}")

    # Expand bounding box 45% for natural context (hair, neck, shoulders hint)
    pad = int(max(w, h) * 0.45)
    ih, iw = img_cv.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(iw, x + w + pad), min(ih, y + h + pad)

    face_crop = img_cv[y1:y2, x1:x2]

    # Pad to square
    size = max(face_crop.shape[:2])
    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - face_crop.shape[0]) // 2
    x_off = (size - face_crop.shape[1]) // 2
    square[y_off:y_off + face_crop.shape[0], x_off:x_off + face_crop.shape[1]] = face_crop

    # Resize to LivePortrait's expected input
    resized = cv2.resize(square, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Slight sharpening — helps LivePortrait produce crisper results
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    resized = cv2.filter2D(resized, -1, kernel)
    resized = np.clip(resized, 0, 255).astype(np.uint8)

    output_path = job_dir / "face_preprocessed.png"
    cv2.imwrite(str(output_path), resized)
    log.info(f"[FACE] Preprocessed face: {output_path}")
    return output_path
