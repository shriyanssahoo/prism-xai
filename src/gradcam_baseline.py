"""
src/gradcam_baseline.py

Generates Grad-CAM heatmaps using the pytorch-grad-cam library, for
side-by-side comparison against PRISM's output (paper's Figures 6,7,11).

Uses the well-maintained pytorch-grad-cam package (Jacob Gildenblat et al.)
rather than a from-scratch reimplementation - properly cited in README.
"""
import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_gradcam(model, target_layer, image_tensor, original_pil_img, target_class=None):
    """
    model: the loaded CNN (e.g. VGG-16)
    target_layer: the conv layer to compute Grad-CAM against (same layer
        used for PRISM, for a fair comparison)
    image_tensor: preprocessed (normalized) single image tensor, shape (3, 224, 224)
    original_pil_img: the original, un-normalized PIL image (for overlay display)
    target_class: ImageNet class index to explain. If None, uses the
        model's own top predicted class for this image.

    Returns:
        cam_overlay: np.ndarray (H, W, 3) uint8 - Grad-CAM heatmap blended on the image
        predicted_class_idx: the class index Grad-CAM explained
    """
    device = next(model.parameters()).device
    input_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dim

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
        target_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(target_class)]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # first (only) image in this batch

    # prepare original image as float [0,1] RGB, resized to match CAM output
    rgb_img = np.array(original_pil_img.convert("RGB").resize((224, 224))) / 255.0
    rgb_img = rgb_img.astype(np.float32)

    cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_overlay, target_class