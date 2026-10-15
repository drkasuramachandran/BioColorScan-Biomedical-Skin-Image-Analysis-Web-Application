"""
Deep-learning models for the Skin Imaging Research Platform (PyTorch only).

Three models live in the ``models/`` folder (or in a Hugging Face model repo):

    my_5layer_cnn.h5    5-layer CNN, 128x128, benign / malignant (sigmoid)
                        -> weights are read straight from the Keras H5 file
                           and loaded into an equivalent PyTorch network,
                           so TensorFlow is NOT required.
    unet_best.pth       U-Net lesion segmentation (256x256)
    densenet_best.pth   DenseNet-121 benign / malignant classifier (224x224)

Model location, in order of priority:
    1. environment variable MODEL_DIR (a local folder)
    2. ./models next to the app
    3. if a file is missing and HF_MODEL_REPO is set (e.g. "user/skin-models"),
       it is downloaded from that Hugging Face model repo.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models as tv_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))

CNN_FILE = "my_5layer_cnn.h5"
UNET_FILE = "unet_best.pth"
DENSENET_FILE = "densenet_best.pth"

CNN_SIZE = 128
UNET_SIZE = 256
CLASSIFIER_SIZE = 224


# ------------------------------------------------------------------
# Model file lookup
# ------------------------------------------------------------------

def _is_lfs_pointer(path: Path) -> bool:
    """True if the file is a tiny Git-LFS pointer instead of real weights."""
    try:
        if path.stat().st_size > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _setting(name: str) -> str:
    """Read a setting from environment variables, else from Streamlit secrets."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def resolve_model_path(filename: str):
    """Return a usable local path for a model file, or None if unavailable."""
    local = MODEL_DIR / filename

    if local.exists() and not _is_lfs_pointer(local):
        return local

    repo = _setting("HF_MODEL_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    token=_setting("HF_TOKEN") or None,
                )
            )
        except Exception:
            return None

    return None


# ------------------------------------------------------------------
# 1. Five-layer CNN (ported from Keras)
# ------------------------------------------------------------------

class FiveLayerCNN(nn.Module):
    """
    Same network as the Keras model:
    Conv32 - Pool - Conv64 - Pool - Conv128 - Pool - Flatten -
    Dense128 - Dropout(0.5) - Dense1(sigmoid).   Input: 128x128x3.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(14 * 14 * 128, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # Keras flattens in (height, width, channel) order, so match it.
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        x = self.drop(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))


_H5_PATHS = {
    "conv1": "model_weights/conv2d/sequential/conv2d",
    "conv2": "model_weights/conv2d_1/sequential/conv2d_1",
    "conv3": "model_weights/conv2d_2/sequential/conv2d_2",
    "fc1": "model_weights/dense/sequential/dense",
    "fc2": "model_weights/dense_1/sequential/dense_1",
}


def load_cnn_from_h5(h5_path) -> FiveLayerCNN:
    import h5py

    model = FiveLayerCNN()
    state = {}

    with h5py.File(h5_path, "r") as f:
        for layer, base in _H5_PATHS.items():
            for suffix in ("kernel", "bias"):
                if f"{base}/{suffix}" not in f:
                    raise RuntimeError(
                        f"Weight missing in H5 file: {base}/{suffix}"
                    )

            kernel = np.array(f[f"{base}/kernel"], dtype=np.float32)
            bias = np.array(f[f"{base}/bias"], dtype=np.float32)

            if layer.startswith("conv"):
                # Keras (kh, kw, in, out) -> PyTorch (out, in, kh, kw)
                kernel = kernel.transpose(3, 2, 0, 1)
            else:
                # Keras (in, out) -> PyTorch (out, in)
                kernel = kernel.T

            state[f"{layer}.weight"] = torch.from_numpy(np.ascontiguousarray(kernel))
            state[f"{layer}.bias"] = torch.from_numpy(bias)

    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def predict_cnn(model, image: Image.Image):
    """Return (malignant_probability, 'Benign' | 'Malignant')."""
    img = image.convert("RGB").resize((CNN_SIZE, CNN_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = float(model(tensor).item())

    return prob, ("Malignant" if prob >= 0.5 else "Benign")


# ------------------------------------------------------------------
# 2. U-Net (segmentation)
# ------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # attribute name "conv" matches the trained checkpoint
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # attribute name "output" matches the trained checkpoint
        self.output = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


def _load_state(model, path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    return model.to(DEVICE).eval()


def load_unet(path):
    return _load_state(UNet(n_channels=3, n_classes=1), path)


# ------------------------------------------------------------------
# 3. DenseNet-121 (classification) + Grad-CAM
# ------------------------------------------------------------------

def load_densenet(path):
    model = tv_models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 0 benign, 1 malignant
    return _load_state(model, path)


_unet_tf = transforms.Compose([
    transforms.Resize((UNET_SIZE, UNET_SIZE)),
    transforms.ToTensor(),
])

_clf_tf = transforms.Compose([
    transforms.Resize((CLASSIFIER_SIZE, CLASSIFIER_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def segment_lesion(model, image):
    tensor = _unet_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    return prob, (prob > 0.5).astype(np.uint8)


def _mask_to_full_res(image, mask):
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST
    )
    return np.array(m) > 127


def extract_lesion(image, mask):
    arr = np.array(image)
    return Image.fromarray((arr * _mask_to_full_res(image, mask)[..., None]).astype(np.uint8))


def create_overlay(image, mask):
    arr = np.array(image).astype(np.float32)
    m = _mask_to_full_res(image, mask)
    arr[m] = arr[m] * 0.5 + np.array([255, 0, 0]) * 0.5
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def classify_lesion(model, image):
    tensor = _clf_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    benign, malignant = float(p[0]), float(p[1])
    return ("Malignant" if malignant >= benign else "Benign"), benign, malignant


def generate_gradcam(model, image):
    """Grad-CAM on the last DenseNet block. Hooks are always removed afterwards."""
    tensor = _clf_tf(image).unsqueeze(0).to(DEVICE)
    layer = model.features.denseblock4
    store = {}

    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("grad", go[0])
    )

    try:
        model.zero_grad()
        output = model(tensor)
        class_index = int(torch.argmax(output, dim=1).item())
        output[0, class_index].backward()

        grads = store["grad"].detach().cpu().numpy()[0]
        acts = store["act"].detach().cpu().numpy()[0]
    finally:
        h1.remove()
        h2.remove()

    weights = grads.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0).astype(np.float32)
    if cam.max() > 0:
        cam /= cam.max()

    return cam, class_index
