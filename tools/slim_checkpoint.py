"""
Make smaller copies of your .pth checkpoints that contain ONLY the model weights
(training extras such as optimizer state are dropped). Originals are never changed.

    python tools/slim_checkpoint.py  models  models_slim

Then upload / use the files from models_slim. The app loads them exactly the same way.
"""
import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

for name in ("unet_best.pth", "densenet_best.pth"):
    f = src / name
    if not f.exists() or f.stat().st_size < 1024:
        print(f"SKIP {name}: not found or looks like an LFS pointer")
        continue

    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    keys = list(ckpt.keys())[:6] if isinstance(ckpt, dict) else "plain model object"
    print(f"{name}: top-level contents -> {keys}")

    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict"):
            if k in ckpt:
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        print(f"  SKIP: unexpected format ({type(ckpt).__name__}), left unchanged")
        continue

    torch.save(ckpt, dst / name)
    print(f"  {f.stat().st_size / 1e6:.0f} MB  ->  {(dst / name).stat().st_size / 1e6:.0f} MB")
