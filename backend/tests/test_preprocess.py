import io

import numpy as np
import torch
from PIL import Image

from main import IMAGE_SIZE, preprocess


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_returns_correctly_shaped_tensor():
    image = Image.new("L", (280, 280), color=0)
    tensor = preprocess(_png_bytes(image))

    assert tensor.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_preprocess_normalizes_pixels_to_unit_interval():
    image = Image.new("L", (280, 280), color=255)
    tensor = preprocess(_png_bytes(image))

    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_preprocess_inverts_light_background_to_dark():
    # A mostly-white image (bright stroke would be invisible) should be
    # flipped so the background reads as dark, matching Kaggle/MNIST pixels.
    light_bg = Image.new("L", (280, 280), color=240)
    tensor = preprocess(_png_bytes(light_bg))

    assert tensor.mean().item() < 0.5


def test_preprocess_leaves_dark_background_unchanged():
    dark_bg = Image.new("L", (280, 280), color=15)
    tensor = preprocess(_png_bytes(dark_bg))

    # 15 / 255 without inversion
    expected = 15.0 / 255.0
    assert np.isclose(tensor.mean().item(), expected, atol=1e-3)


def test_preprocess_converts_color_images_to_grayscale():
    color_image = Image.new("RGB", (280, 280), color=(10, 20, 30))
    tensor = preprocess(_png_bytes(color_image))

    assert tensor.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE)
