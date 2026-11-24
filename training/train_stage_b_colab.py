"""Magnum Stage B: train a deployment ResNet18 on all 3,723 quality-filtered images.

Usage in the CURRENT restored Colab notebook:
    from google.colab import files
    uploaded = files.upload()  # choose this .py only
    from pathlib import Path
    Path('/content/Magnum_StageB_train_full_quality.py').write_bytes(
        uploaded['Magnum_StageB_train_full_quality.py'])
    %run -i /content/Magnum_StageB_train_full_quality.py

After training, a single ZIP is offered for download.
Do not reuse the Stage A validation/test metrics for this new checkpoint.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

SEED_B = 42
CLASSES_B = ['Capsicum', 'Carrot', 'Cucumber', 'Potato', 'Raddish', 'Squash']
MEAN_B = [0.485, 0.456, 0.406]
STD_B = [0.229, 0.224, 0.225]
CFG_B = {
    'seed': 42,
    'batch_size': 32,
    'num_workers': 2,
    'head_epochs': 2,
    'fine_epochs': 3,
    'total_epochs': 5,
    'lr_head': 1e-3,
    'lr_layer4': 1e-4,
    'lr_fc_fine': 3e-4,
    'weight_decay': 1e-4,
    'image_size': 224,
}
STAGE_B_ROOT = Path('/content/Magnum_StageB')
STAGE_B_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_B = STAGE_B_ROOT / 'resnet18_stage_b_full_quality_epoch5.pth'

print('=== MAGNUM STAGE B: ФИНАЛЬНОЕ ОБУЧЕНИЕ ===', flush=True)
assert torch.cuda.is_available(), 'Для обучения нужен GPU: Colab -> Среда выполнения -> T4 GPU'
print('GPU:', torch.cuda.get_device_name(0), flush=True)

# Reuse restored CSV / DataFrame without uploading it again if available.
if 'split_df' in globals() and isinstance(globals()['split_df'], pd.DataFrame):
    stage_b_manifest = globals()['split_df'].copy()
    print('CSV: использую split_df из текущей сессии Colab.', flush=True)
else:
    csv_path = Path('/content/stage_a_split_v3.csv')
    if not csv_path.is_file():
        from google.colab import files
        print('Загрузи только stage_a_split_v3.csv:', flush=True)
        uploaded = files.upload()
        assert csv_path.name in uploaded, 'Не найден stage_a_split_v3.csv'
        csv_path.write_bytes(uploaded[csv_path.name])
    stage_b_manifest = pd.read_csv(csv_path)
    print('CSV:', csv_path, flush=True)

required_columns = {'class', 'filename', 'split', 'split_group', 'use_quality_only'}
assert required_columns.issubset(stage_b_manifest.columns), 'В CSV отсутствуют обязательные столбцы'
assert len(stage_b_manifest) == 3723, f'Ожидалось 3723 фото, найдено {len(stage_b_manifest)}'
assert stage_b_manifest['use_quality_only'].astype(str).str.lower().eq('true').all(), 'Найдены фото вне Quality'
assert set(stage_b_manifest['split']) == {'train', 'val', 'test'}, 'Некорректная структура разбиения V3'
assert stage_b_manifest['split_group'].nunique() == 202, 'Неожиданное количество групп съёмок'
assert not stage_b_manifest.duplicated(['class','filename']).any(), 'Дубликат строки class + filename'
assert sorted(stage_b_manifest['class'].unique()) == CLASSES_B, 'Не совпадают 6 классов'
assert stage_b_manifest['split'].value_counts().to_dict() == {'train':2488,'test':683,'val':552}

# Reuse Kaggle photos if the Stage A session is still alive; otherwise fetch version 1.
source_candidate = globals().get('SOURCE_ROOT', None)
if source_candidate is not None and Path(source_candidate).is_dir():
    source_root_b = Path(source_candidate)
else:
    try:
        import kagglehub
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'kagglehub'])
        import kagglehub
    dataset_path_b = Path(kagglehub.dataset_download('preethi866/vegetable-dataset/versions/1'))
    source_root_b = dataset_path_b / 'Dataset' / 'Dataset'
assert source_root_b.is_dir(), f'Исходные фото не найдены: {source_root_b}'

stage_b_manifest['image_path'] = [
    str(source_root_b / c / f)
    for c, f in zip(stage_b_manifest['class'], stage_b_manifest['filename'])
]
missing_paths = [p for p in stage_b_manifest['image_path'] if not Path(p).is_file()]
assert not missing_paths, f'Нет {len(missing_paths)} фото, первый: {missing_paths[0]}'

stage_b_manifest.to_csv(STAGE_B_ROOT / 'stage_b_full_quality_manifest.csv', index=False)
class_counts = stage_b_manifest['class'].value_counts().reindex(CLASSES_B)
assert (class_counts > 0).all()
print('\nСостав финального Train:')
print(class_counts.to_string())
print('Всего:', len(stage_b_manifest))
print('Уникальных съёмочных групп:', stage_b_manifest['split_group'].nunique())
print('Источник:', source_root_b)
print('Источники строк (старое разбиение):', stage_b_manifest['split'].value_counts().to_dict())
print('Важно: старые Validation/Test ТЕПЕРЬ входят в Train B.')
print('У финальной модели B нет независимой Test-оценки.')
print('План: 2 эпохи fc + 3 эпохи layer4/fc; расписание выбрано по Stage A.\n')

# Identical preprocessing and training recipe to Stage A.
train_transform_b = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_B, STD_B),
])
infer_transform_b = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_B, STD_B),
])

class StageBDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.labels = {name: i for i, name in enumerate(CLASSES_B)}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row['image_path']) as im:
            x = train_transform_b(ImageOps.exif_transpose(im).convert('RGB'))
        return x, self.labels[row['class']]


def seed_everything_b(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker_b(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything_b(SEED_B)
generator_b = torch.Generator().manual_seed(SEED_B)
train_loader_b = DataLoader(
    StageBDataset(stage_b_manifest),
    batch_size=CFG_B['batch_size'],
    shuffle=True,
    num_workers=CFG_B['num_workers'],
    pin_memory=True,
    worker_init_fn=seed_worker_b,
    generator=generator_b,
)

model_b = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_b.fc = nn.Linear(model_b.fc.in_features, len(CLASSES_B))
model_b = model_b.to('cuda')
weights_b = len(stage_b_manifest) / (len(CLASSES_B) * class_counts.to_numpy())
criterion_b = nn.CrossEntropyLoss(
    weight=torch.tensor(weights_b, dtype=torch.float32, device='cuda')
)
scaler_b = torch.amp.GradScaler('cuda')
history_b = []
started_b = time.time()

for epoch_b in range(1, CFG_B['total_epochs']+1):
    if epoch_b == 1:
        for p in model_b.parameters():
            p.requires_grad = False
        for p in model_b.fc.parameters():
            p.requires_grad = True
        optimizer_b = torch.optim.AdamW(
            model_b.fc.parameters(),
            lr=CFG_B['lr_head'],
            weight_decay=CFG_B['weight_decay'],
        )
        phase_b = 'head'
    elif epoch_b == CFG_B['head_epochs']+1:
        for p in model_b.layer4.parameters():
            p.requires_grad = True
        optimizer_b = torch.optim.AdamW([
            {'params': model_b.layer4.parameters(), 'lr': CFG_B['lr_layer4']},
            {'params': model_b.fc.parameters(), 'lr': CFG_B['lr_fc_fine']},
        ], weight_decay=CFG_B['weight_decay'])
        phase_b = 'fine_tune'

    model_b.train()
    for module_b in model_b.modules():
        if isinstance(module_b, nn.BatchNorm2d):
            if not any(p.requires_grad for p in module_b.parameters()):
                module_b.eval()

    total_b = 0
    hits_b = 0
    loss_sum_b = 0.0
    print(f'\nЭпоха B {epoch_b}/{CFG_B["total_epochs"]} — {phase_b}', flush=True)
    for images_b, labels_b in tqdm(train_loader_b, desc=f'Stage B epoch {epoch_b}', leave=False):
        images_b = images_b.to('cuda', non_blocking=True)
        labels_b = labels_b.to('cuda', non_blocking=True)
        optimizer_b.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits_b = model_b(images_b)
            loss_b = criterion_b(logits_b, labels_b)
        scaler_b.scale(loss_b).backward()
        scaler_b.step(optimizer_b)
        scaler_b.update()
        bs_b = labels_b.size(0)
        total_b += bs_b
        loss_sum_b += float(loss_b.item()) * bs_b
        hits_b += int((logits_b.argmax(1) == labels_b).sum().item())

    avg_train_loss_b = loss_sum_b / total_b
    avg_train_acc_b = hits_b / total_b
    history_b.append({
        'epoch': epoch_b,
        'phase': phase_b,
        'train_loss': avg_train_loss_b,
        'train_accuracy': avg_train_acc_b,
        'n_train': total_b,
    })
    print(f'Train loss={avg_train_loss_b:.5f} | Train accuracy={avg_train_acc_b:.4f}', flush=True)

    # Keep latest epoch in runtime as recovery protection; never select on old Test.
    torch.save({
        'architecture': 'resnet18',
        'weights_source': 'IMAGENET1K_V1',
        'model_state_dict': model_b.state_dict(),
        'class_names': CLASSES_B,
        'experiment': 'stage_b_full_quality',
        'epoch': epoch_b,
        'config': CFG_B.copy(),
        'mean': MEAN_B,
        'std': STD_B,
        'class_counts': class_counts.astype(int).tolist(),
        'seed': SEED_B,
        'torch_version': str(torch.__version__),
        'data': {
            'train_size': int(len(stage_b_manifest)),
            'split_group_count': int(stage_b_manifest['split_group'].nunique()),
            'original_splits_combined': ['train', 'val', 'test'],
            'has_independent_holdout': False,
        },
    }, CHECKPOINT_B)
    pd.DataFrame(history_b).to_csv(STAGE_B_ROOT/'stage_b_epoch_history.csv', index=False)
    print('Checkpoint сохранён:', CHECKPOINT_B, flush=True)

# CPU smoke test, no independent accuracy claims.
checkpoint_on_cpu_b = torch.load(CHECKPOINT_B, map_location='cpu', weights_only=True)
cpu_model_b = resnet18(weights=None)
cpu_model_b.fc = nn.Linear(cpu_model_b.fc.in_features, len(CLASSES_B))
cpu_model_b.load_state_dict(checkpoint_on_cpu_b['model_state_dict'], strict=True)
cpu_model_b.eval()
example_b = Path(stage_b_manifest.iloc[0]['image_path'])
with Image.open(example_b) as img_b:
    test_tensor_b = infer_transform_b(ImageOps.exif_transpose(img_b).convert('RGB')).unsqueeze(0)
with torch.inference_mode():
    cpu_probs_b = torch.softmax(cpu_model_b(test_tensor_b), dim=1)[0]
cpu_pred_b = CLASSES_B[int(cpu_probs_b.argmax())]
assert cpu_probs_b.shape == (6,)

run_seconds_b = round(time.time()-started_b, 2)
summary_b = {
    'experiment': 'stage_b_full_quality',
    'strategy_from_stage_a': 'quality_only (not CNN Diversity)',
    'train_size': int(len(stage_b_manifest)),
    'split_group_count': int(stage_b_manifest['split_group'].nunique()),
    'class_counts': class_counts.astype(int).to_dict(),
    'training_schedule': '2 head epochs + 3 fine-tuning epochs; fixed using Stage A epoch 5',
    'validation_used_in_stage_b': False,
    'test_used_for_stage_b_selection': False,
    'independent_evaluation_of_stage_b': False,
    'stage_a_metrics_must_not_be_attributed_to_stage_b': True,
    'seed': SEED_B,
    'config': CFG_B,
    'cpu_smoke_test': {'success': True, 'image': str(example_b), 'predicted_class': cpu_pred_b,
                       'scores': {c: round(float(p),6) for c,p in zip(CLASSES_B,cpu_probs_b.tolist())}},
    'runtime_seconds': run_seconds_b,
    'torch_version': str(torch.__version__),
}
(STAGE_B_ROOT / 'stage_b_summary.json').write_text(
    json.dumps(summary_b, indent=2, ensure_ascii=False), encoding='utf-8')
archive_b = STAGE_B_ROOT / 'Magnum_StageB_full_quality_results.zip'
with zipfile.ZipFile(archive_b,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in ['resnet18_stage_b_full_quality_epoch5.pth',
                 'stage_b_epoch_history.csv',
                 'stage_b_full_quality_manifest.csv',
                 'stage_b_summary.json']:
        z.write(STAGE_B_ROOT/name, arcname=name)

print('\n=== STAGE B ЗАВЕРШЕНА ===')
print('Использовано 3723 фото из 202 съёмочных групп.')
print('Ровно 5 эпох, финальная модель сохранена.')
print('CPU-проверка: OK, предсказание:', cpu_pred_b)
print('Отдельной независимой метрики Test у модели B нет.')
print('Архив:', archive_b)
from google.colab import files
files.download(str(archive_b))
