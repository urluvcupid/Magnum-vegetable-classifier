#!/usr/bin/env python3
"""Evaluate the frozen Magnum ResNet18 on the original split V3, without training."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

from predict import build_predictor, load_checkpoint

CLASSES = ['Capsicum', 'Carrot', 'Cucumber', 'Potato', 'Raddish', 'Squash']


class Images(Dataset):
    def __init__(self, frame, data_root, preprocess):
        self.frame = frame.reset_index(drop=True)
        self.root = data_root
        self.preprocess = preprocess
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = self.root / row['class'] / row['filename']
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            x = self.preprocess(rgb)
        return x, CLASSES.index(row['class'])


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Reproduce validation/test metrics using split V3')
    p.add_argument('--data-root', type=Path, required=True, help='Kaggle Dataset/Dataset root')
    p.add_argument('--split-csv', type=Path, default=base/'split_manifests/stage_a_split_v3.csv')
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--weights', type=Path, default=base/'model/resnet18_quality_best.pth',
                   help='Only Stage A checkpoint is valid for old val/test')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--out-dir', type=Path, default=base/'evaluation_reproduced')
    args = p.parse_args()
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device(device_name)
    frame = pd.read_csv(args.split_csv)
    frame = frame[frame['split']==args.split].copy().reset_index(drop=True)
    if not len(frame):
        raise ValueError('Selected split is empty')
    checkpoint = load_checkpoint(args.weights)
    if checkpoint.get('experiment') == 'stage_b_full_quality':
        raise ValueError('Stage B was trained on original val/test and cannot be evaluated as an independent holdout.')
    model, preprocess, checkpoint_classes = build_predictor(args.weights, device)
    assert list(checkpoint_classes) == CLASSES
    data = Images(frame, args.data_root, preprocess)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type=='cuda'))
    preds, targets, scores = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits.float(), dim=1).cpu()
            values, indices = probs.max(dim=1)
            preds.extend(indices.tolist())
            scores.extend(values.tolist())
            targets.extend(labels.tolist())
    accuracy = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, labels=list(range(6)), average='macro', zero_division=0))
    out = args.out_dir / args.split
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(split=args.split, n=len(frame), accuracy=accuracy,
                   macro_f1=macro_f1, correct=sum(a==b for a,b in zip(targets,preds)),
                   model=str(args.weights), device=device_name)
    (out/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame(classification_report(targets, preds, labels=list(range(6)),
                                       target_names=CLASSES, output_dict=True,
                                       zero_division=0)).T.to_csv(out/'classification_report.csv')
    pd.DataFrame(confusion_matrix(targets, preds, labels=list(range(6))),
                 index=CLASSES, columns=CLASSES).to_csv(out/'confusion_matrix.csv')
    frame['predicted_class'] = [CLASSES[i] for i in preds]
    frame['softmax_score'] = scores
    frame['correct'] = [a==b for a,b in zip(targets,preds)]
    frame.to_csv(out/'predictions.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Saved results:', out)
    if device.type=='cpu':
        print('NOTE: Original Colab test used CUDA AMP; tiny numerical variations on CPU are possible.')


if __name__=='__main__':
    main()
