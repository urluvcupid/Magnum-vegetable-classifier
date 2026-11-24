#!/usr/bin/env python3
"""Offline CPU ResNet18 inference for Magnum.

Required employer interface:
    python predict.py --input /path/to/images --output preds.csv

Optional legacy single-image interface:
    python predict.py --image /path/to/photo.jpg [--json]
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.torch_version import TorchVersion
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "model" / "resnet18_stage_b_full_quality_epoch5.pth"
EXPECTED_CLASSES = ("Capsicum", "Carrot", "Cucumber", "Potato", "Raddish", "Squash")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def load_checkpoint(path: Path):
    """Read local weights with no Internet access."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_predictor(weights_path: Path, device: torch.device):
    checkpoint = load_checkpoint(Path(weights_path))
    classes = tuple(checkpoint["class_names"])
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Expected ResNet18 checkpoint")

    model = resnet18(weights=None)  # No pretrained-weight download.
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    size = int(checkpoint.get("config", {}).get("image_size", 224))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, preprocess, classes


def prepare_image(image_path: Path, preprocess):
    with Image.open(image_path) as image:
        return preprocess(ImageOps.exif_transpose(image).convert("RGB"))


def infer_one(model, preprocess, classes, image_path: Path, device: torch.device):
    tensor = prepare_image(image_path, preprocess).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
    idx = max(range(len(probs)), key=lambda index: probs[index])
    return {"filename": image_path.name, "label": classes[idx],
            "confidence": float(probs[idx]),
            "scores": {name: float(prob) for name, prob in zip(classes, probs)}}


def choose_device(name: str):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu")
    return torch.device(name)


def predict(image_path: Path, weights_path: Path = DEFAULT_WEIGHTS, device_name: str = "cpu"):
    """Legacy programmatic single-image interface."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    device = choose_device(device_name)
    model, preprocess, classes = build_predictor(weights_path, device)
    result = infer_one(model, preprocess, classes, image_path, device)
    return {"image": str(image_path), "predicted_class": result["label"],
            "confidence": result["confidence"], "device": str(device),
            "scores": result["scores"]}


def predict_directory(input_dir: Path, output_csv: Path,
                      weights_path: Path = DEFAULT_WEIGHTS, device_name: str = "cpu"):
    """Process a folder with one model load, writing exactly the required columns."""
    input_dir, output_csv = Path(input_dir), Path(output_csv)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    paths = sorted(
        (p for p in input_dir.rglob("*")
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(input_dir).as_posix().casefold(),
    )
    if not paths:
        raise ValueError(f"No supported images in {input_dir}")

    device = choose_device(device_name)
    model, preprocess, classes = build_predictor(weights_path, device)
    rows = []
    for path in paths:
        try:
            result = infer_one(model, preprocess, classes, path, device)
        except Exception as exc:
            raise RuntimeError(f"Failed to predict {path}: {exc}") from exc
        # Preserve relative paths for identically named files in subfolders.
        rows.append({"filename": path.relative_to(input_dir).as_posix(),
                     "label": result["label"],
                     "confidence": f"{result['confidence']:.6f}"})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Magnum: offline six-vegetable classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Directory of images (required by employer)")
    source.add_argument("--image", type=Path, help="Optional single-image demonstration")
    parser.add_argument("--output", type=Path, help="Output CSV (required with --input)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--json", action="store_true", help="JSON for --image mode")
    args = parser.parse_args()
    if args.input is not None:
        if args.output is None:
            parser.error("--output is required when using --input")
        n = predict_directory(args.input, args.output, args.weights, args.device)
        print(f"Saved {n} predictions to {args.output}")
    else:
        if args.output is not None:
            parser.error("--output is only for --input directory mode")
        result = predict(args.image, args.weights, args.device)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence (softmax, uncalibrated): {result['confidence']:.6f}")
            print(f"Device: {result['device']}")
            for name, score in sorted(result["scores"].items(), key=lambda row: -row[1]):
                print(f"  {name}: {score:.6f}")


if __name__ == "__main__":
    main()
