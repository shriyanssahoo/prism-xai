"""
Utilities for loading pretrained CNNs and extracting intermediate
layer activations via forward hooks.
Adapted conceptually from the public TorchPRISM repo (Szandała, 2022):
https://github.com/szandala/TorchPRISM  -- reimplemented independently for this project.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image(path):
    """Load an image file and return the preprocessed tensor + original PIL image."""
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img)
    return tensor, img


def load_model(name="vgg16"):
    """Load a pretrained torchvision model in eval mode."""
    if name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif name == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unsupported model: {name}")
    model.eval()
    model.to(DEVICE)
    return model


class ActivationExtractor:
    """
    Registers a forward hook on a target layer and stores its output
    (the 4D representation batch_size x channels x H x W) for PRISM.
    """
    def __init__(self, model, layer):
        self.activations = None
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations = output.detach()

    def remove(self):
        self.hook.remove()


def get_vgg16_last_conv_layer(model):
    """VGG16's features block ends with layer index 29 (last conv before final pool)."""
    return model.features[28]  # last Conv2d before the final MaxPool


def batch_predict(model, image_tensors):
    """Run a batch of preprocessed image tensors through the model, return softmax probs."""
    batch = torch.stack(image_tensors).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    return probs