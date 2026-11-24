#!/usr/bin/env python3
"""Reproduce the Stage A train/validation comparison using Kaggle's original photos.

Never uses Test for model selection or hyperparameter tuning.
Example:
  python training/train_stage_a.py --data-root /kaggle/input/vegetable-dataset/Dataset/Dataset --experiment both

The exact saved checkpoint is already provided in ../model and does NOT need retraining for inference.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm.auto import tqdm

CLASSES = ['Capsicum', 'Carrot', 'Cucumber', 'Potato', 'Raddish', 'Squash']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CFG = dict(seed=42, batch_size=32, num_workers=2, head_epochs=2, fine_epochs=5,
           lr_head=1e-3, lr_layer4=1e-4, lr_fc_fine=3e-4, weight_decay=1e-4,
           image_size=224)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class VegDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r['image_path']) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            x = self.transform(im)
        return x, CLASSES.index(r['class'])


def add_paths(df, root):
    df = df.copy()
    df['image_path'] = [str(root / c / f) for c, f in zip(df['class'], df['filename'])]
    missing = [p for p in df['image_path'] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} files; first: {missing[0]}')
    return df


def make_loaders(train_df, val_df):
    gen = torch.Generator().manual_seed(CFG['seed'])
    train = DataLoader(VegDataset(train_df, train_transform), batch_size=CFG['batch_size'],
                       shuffle=True, num_workers=CFG['num_workers'], pin_memory=True,
                       worker_init_fn=seed_worker, generator=gen)
    val = DataLoader(VegDataset(val_df, val_transform), batch_size=CFG['batch_size'],
                     shuffle=False, num_workers=CFG['num_workers'], pin_memory=True)
    return train, val


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
    loss_sum, hits, total = 0.0, 0, 0
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images, targets = images.to('cuda', non_blocking=True), targets.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)
        hits += (logits.argmax(1) == targets).sum().item()
    return loss_sum / total, hits / total


@torch.inference_mode()
def validate(model, loader):
    model.eval()
    true, pred, loss_sum, total = [], [], 0.0, 0
    loss_fn = nn.CrossEntropyLoss()
    for images, targets in loader:
        images, targets = images.to('cuda', non_blocking=True), targets.to('cuda', non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, targets)
        total += targets.size(0)
        loss_sum += loss.item() * targets.size(0)
        true.extend(targets.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    return dict(val_loss=loss_sum / total,
                val_accuracy=float(accuracy_score(true, pred)),
                val_macro_f1=float(f1_score(true, pred, labels=list(range(6)),
                                          average='macro', zero_division=0)))


def run(train_df, val_df, name, output_dir):
    set_seed(CFG['seed'])
    train_loader, val_loader = make_loaders(train_df, val_df)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to('cuda')
    class_counts = train_df['class'].value_counts().reindex(CLASSES).to_numpy()
    assert np.all(class_counts > 0)
    class_weights = torch.tensor(len(train_df) / (6 * class_counts),
                                 dtype=torch.float32, device='cuda')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')
    history, best, best_epoch = [], (-1., -1.), None
    checkpoint_path = output_dir / f'resnet18_{name}_best.pth'
    started = time.time()
    for epoch in range(1, CFG['head_epochs'] + CFG['fine_epochs'] + 1):
        if epoch == 1:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=CFG['lr_head'],
                                          weight_decay=CFG['weight_decay'])
            phase = 'head'
        if epoch == CFG['head_epochs'] + 1:
            for p in model.layer4.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                dict(params=model.layer4.parameters(), lr=CFG['lr_layer4']),
                dict(params=model.fc.parameters(), lr=CFG['lr_fc_fine']),
            ], weight_decay=CFG['weight_decay'])
            phase = 'fine_tune'
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler)
        metrics = validate(model, val_loader)
        history.append(dict(epoch=epoch, phase=phase, train_loss=tr_loss,
                            train_accuracy=tr_acc, **metrics))
        print(f'{name} epoch {epoch}: accuracy={metrics["val_accuracy"]:.4f}, '
              f'macro-F1={metrics["val_macro_f1"]:.4f}')
        key = (metrics['val_macro_f1'], metrics['val_accuracy'])
        if key > best:
            best, best_epoch = key, epoch
            torch.save(dict(
                architecture='resnet18', weights_source='IMAGENET1K_V1',
                model_state_dict=model.state_dict(), class_names=CLASSES,
                experiment=name, epoch=epoch, validation=metrics, config=CFG.copy(),
                mean=MEAN, std=STD, class_counts=class_counts.tolist(),
                seed=CFG['seed'], torch_version=torch.__version__
            ), checkpoint_path)
    pd.DataFrame(history).to_csv(output_dir / f'resnet18_{name}_history.csv', index=False)
    summary = dict(experiment=name, train_size=len(train_df), validation_size=len(val_df),
                   best_epoch=best_epoch, val_macro_f1=best[0], val_accuracy=best[1],
                   runtime_minutes=(time.time()-started)/60, settings=CFG)
    (output_dir / f'resnet18_{name}_config.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', checkpoint_path)


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True,
                   help='Path to original Kaggle Dataset/Dataset containing class directories')
    p.add_argument('--split-csv', type=Path, default=root/'split_manifests/stage_a_split_v3.csv')
    p.add_argument('--cnn-csv', type=Path, default=root/'split_manifests/stage_a_train_cnn_balanced_v3.csv')
    p.add_argument('--output-dir', type=Path, default=root/'training_outputs')
    p.add_argument('--experiment', choices=['both','quality','cnn_diversity'], default='both')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA GPU; inference predict.py works on CPU.')
    all_split = pd.read_csv(args.split_csv)
    cnn_split = pd.read_csv(args.cnn_csv)
    val = add_paths(all_split[all_split['split'] == 'val'], args.data_root)
    quality = add_paths(all_split[all_split['split'] == 'train'], args.data_root)
    cnn = add_paths(cnn_split, args.data_root)
    assert len(quality) == 2488 and len(cnn) == 1798 and len(val) == 552
    assert (cnn['split'] == 'train').all()
    assert set(quality['split_group']).isdisjoint(val['split_group'])
    assert set(cnn['split_group']).isdisjoint(val['split_group'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment in ('both', 'quality'):
        run(quality, val, 'quality', args.output_dir)
    if args.experiment in ('both', 'cnn_diversity'):
        run(cnn, val, 'cnn_diversity', args.output_dir)

if __name__ == '__main__':
    main()
