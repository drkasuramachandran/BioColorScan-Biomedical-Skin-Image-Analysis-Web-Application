"""
Sanity-check the model files and (optionally) the Keras -> PyTorch port.

    python tools/check_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from modules import dl_models as dl

img = Image.fromarray((np.random.rand(300, 400, 3) * 255).astype("uint8"))

for fname in (dl.CNN_FILE, dl.UNET_FILE, dl.DENSENET_FILE):
    p = dl.resolve_model_path(fname)
    print(f"{fname:22s}", p if p else "NOT FOUND (or LFS pointer)")

p = dl.resolve_model_path(dl.CNN_FILE)
if p:
    cnn = dl.load_cnn_from_h5(p)
    prob, label = dl.predict_cnn(cnn, img)
    print("CNN (PyTorch)  :", round(prob, 5), label)
    try:  # compare with the original Keras implementation if TensorFlow is installed
        import tensorflow as tf  # noqa
        sys.path.insert(0, "tools")
        print("TensorFlow found: compare against the original Keras model manually "
              "(same image resized to 128x128 / 255) - values should agree to ~1e-5.")
    except ImportError:
        print("TensorFlow not installed: skipped Keras comparison.")

p = dl.resolve_model_path(dl.UNET_FILE)
if p:
    unet = dl.load_unet(p)
    _, mask = dl.segment_lesion(unet, img)
    print("U-Net mask     :", mask.shape, "lesion fraction", round(float(mask.mean()), 4))

p = dl.resolve_model_path(dl.DENSENET_FILE)
if p:
    net = dl.load_densenet(p)
    print("DenseNet       :", dl.classify_lesion(net, img))
    cam, idx = dl.generate_gradcam(net, img)
    print("Grad-CAM       :", cam.shape, "class", idx)
