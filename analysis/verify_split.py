#!/usr/bin/env python3
"""Verify frozen split V3 and CNN subset using only shipped CSV manifests."""
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/'split_manifests'/'stage_a_split_v3.csv'
CNN=ROOT/'split_manifests'/'stage_a_train_cnn_balanced_v3.csv'
with SPLIT.open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
with CNN.open(newline='',encoding='utf-8') as f:
    cnn=list(csv.DictReader(f))
quality=[r for r in rows if str(r['use_quality_only']).strip().lower() in {'true','1','yes'}]
assert len(rows)==3723==len(quality)
counts={k:sum(r['split']==k for r in quality) for k in ('train','val','test')}
assert counts=={'train':2488,'val':552,'test':683},counts
groups={k:{r['split_group'] for r in quality if r['split']==k} for k in counts}
assert not groups['train']&groups['val']
assert not groups['train']&groups['test']
assert not groups['val']&groups['test']
train_keys={(r['class'],r['filename']) for r in quality if r['split']=='train'}
cnn_keys={(r['class'],r['filename']) for r in cnn}
assert len(cnn_keys)==len(cnn)==1798 and cnn_keys<=train_keys
print('Verified',counts,'CNN Diversity',len(cnn),'distinct group labels',len(set(r['split_group'] for r in quality)))
