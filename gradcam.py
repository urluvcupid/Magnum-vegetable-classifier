#!/usr/bin/env python3
"""Generate a Grad-CAM overlay for one image using the frozen model.

Optional dependencies: numpy, matplotlib. See README for installation.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from predict import DEFAULT_WEIGHTS, build_predictor


def run(image_path: Path, weights_path: Path, output_path: Path, device_name: str):
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна, используй --device cpu')
    device = torch.device(device_name)
    model, preprocess, class_names = build_predictor(weights_path, device)
    model.eval()
    with Image.open(image_path) as img:
        original = ImageOps.exif_transpose(img).convert('RGB').copy()
    original.thumbnail((720, 720))
    x = preprocess(original).unsqueeze(0).to(device).requires_grad_(True)
    captured = {}

    def hook(_module, _inputs, output):
        captured['features'] = output
        output.register_hook(lambda grads: captured.__setitem__('grads', grads))

    handle = model.layer4[-1].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        pred_idx = int(logits.argmax(1).item())
        logits[0, pred_idx].backward()
    finally:
        handle.remove()

    feats = captured['features']
    grads = captured['grads']
    pooled_grads = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((feats * pooled_grads).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(original.height, original.width),
                        mode='bilinear', align_corners=False)
    heatmap = cam.detach().cpu().numpy()[0, 0]
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)
    image_rgb = np.asarray(original, dtype=np.float32) / 255.
    overlay = np.clip(image_rgb * .65 + plt.get_cmap('jet')(heatmap)[..., :3] * .35, 0, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, overlay)
    print(f'Class: {class_names[pred_idx]} | device: {device} | Grad-CAM: {output_path}')
    print('Target = predicted class; this heatmap is not a causal explanation.')


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for a single image')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output', type=Path, default=Path('gradcam_overlay.png'))
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    run(args.image, args.weights, args.output, args.device)


if __name__ == '__main__':
    main()
