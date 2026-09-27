"""
src/visualize.py

Converts PRISM's low-resolution PC maps into human-viewable overlays
by upsampling and blending with the original image.
"""
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def upsample_prism_map(prism_map_single, target_size):
    """
    prism_map_single: (3, H, W) array for ONE image, values in [0,1]
    target_size: (width, height) of the original image to match
    Returns: (target_h, target_w, 3) uint8 RGB array
    """
    # move channel to last dim for cv2: (H, W, 3)
    chw = np.transpose(prism_map_single, (1, 2, 0))
    resized = cv2.resize(chw, target_size, interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    return (resized * 255).astype(np.uint8)


def overlay_prism_on_image(original_pil_img, prism_map_single, alpha=0.5):
    """
    original_pil_img: PIL.Image (original, un-normalized)
    prism_map_single: (3, H, W) array for this one image from prism_maps batch
    alpha: blend strength of the PRISM overlay
    Returns: PIL.Image of the blended result
    """
    orig = np.array(original_pil_img.convert("RGB"))
    target_size = (orig.shape[1], orig.shape[0])  # (width, height) for cv2

    prism_rgb = upsample_prism_map(prism_map_single, target_size)

    blended = cv2.addWeighted(orig, 1 - alpha, prism_rgb, alpha, 0)
    return Image.fromarray(blended)


def plot_prism_batch(original_images, prism_maps, titles=None, save_path=None):
    """
    original_images: list of PIL.Image (original images, one per batch item)
    prism_maps: np.ndarray (batch_size, 3, H, W) from compute_prism()
    titles: optional list of strings for subplot titles (e.g. predicted class + confidence)
    save_path: if given, saves the figure instead of just showing it
    """
    n = len(original_images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        orig = original_images[i]
        overlay = overlay_prism_on_image(orig, prism_maps[i])

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[0, i].set_title(titles[i] if titles else f"Image {i+1}")

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title("PRISM overlay")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    
    
def plot_prism_vs_gradcam(original_images, prism_maps, gradcam_overlays, titles=None, save_path=None):
    """
    Side-by-side comparison: original | Grad-CAM | PRISM
    (mirrors the paper's Figures 6, 7, 11 layout)

    original_images: list of PIL.Image
    prism_maps: np.ndarray (batch_size, 3, H, W) from compute_prism()
    gradcam_overlays: list of np.ndarray (H, W, 3) uint8, one per image
    """
    n = len(original_images)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i in range(n):
        orig = original_images[i]
        prism_overlay = overlay_prism_on_image(orig, prism_maps[i])

        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[0, i].set_title(titles[i] if titles else f"Image {i+1}")

        axes[1, i].imshow(gradcam_overlays[i])
        axes[1, i].axis("off")
        axes[1, i].set_title("Grad-CAM")

        axes[2, i].imshow(prism_overlay)
        axes[2, i].axis("off")
        axes[2, i].set_title("PRISM")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()