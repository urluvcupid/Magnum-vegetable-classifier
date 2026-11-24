#!/usr/bin/env python3
"""Recalculate selected Magnum historical claims from shipped frozen CSV files.

No image download, GPU, or training; no scikit-learn needed.
This does not reconstruct unexported manual image reviews or independently prove real
capture-session independence. It checks recorded manifests, selection, and predictions.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'analysis' / 'research_tables'
SPLITS = ROOT / 'split_manifests'
RESULTS = ROOT / 'results'
CLASSES = ['Capsicum','Carrot','Cucumber','Potato','Raddish','Squash']


def read_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def ids(rows):
    keys = {(r['class'], r['filename']) for r in rows}
    assert len(keys) == len(rows), f'Duplicate (class, filename), {len(rows)=}, {len(keys)=}'
    return keys


def classification(rows):
    true = Counter(r['class'] for r in rows)
    correct = Counter()
    pred = Counter(r['predicted_class'] for r in rows)
    hits = 0
    for r in rows:
        assert r['class'] in CLASSES and r['predicted_class'] in CLASSES
        if r['class'] == r['predicted_class']:
            hits += 1
            correct[r['class']] += 1
        assert (r['correct'].lower() == 'true') == (r['class'] == r['predicted_class'])
    f1 = []
    for c in CLASSES:
        tp = correct[c]
        denom = 2 * tp + (pred[c] - tp) + (true[c] - tp)
        f1.append(2 * tp / denom if denom else 0)
    return {'n':len(rows),'correct':hits,'accuracy':hits/len(rows),
            'macro_f1':sum(f1)/len(CLASSES),'support':dict(true),'per_class_f1':dict(zip(CLASSES,f1))}


def main():
    source = read_rows(EVIDENCE / 'master_manifest.csv')
    quality = read_rows(EVIDENCE / 'quality_flags_df.csv')
    split = read_rows(SPLITS / 'stage_a_split_v3.csv')
    cnn_train = read_rows(SPLITS / 'stage_a_train_cnn_balanced_v3.csv')
    emb_manifest = read_rows(EVIDENCE / 'embedding_manifest.csv')
    old_cnn = read_rows(EVIDENCE / 'cnn_balanced.csv')
    neighbors = read_rows(EVIDENCE / 'neighbor_ssim.csv')
    stage_b = read_rows(RESULTS / 'stage_b' / 'stage_b_full_quality_manifest.csv')
    classes = Counter(r['class'] for r in source)
    assert classes == {'Capsicum':615,'Carrot':905,'Cucumber':308,
                       'Potato':422,'Raddish':286,'Squash':1343}
    originals, flagged = ids(source), ids(quality)
    assert len(originals) == 3879 and len(flagged) == 156
    assert all(r['quality_decision'] == 'exclude_candidate' for r in quality)
    assert len(split) == 3723 and ids(split) == originals - flagged
    assert len(stage_b) == 3723 and ids(stage_b) == ids(split)
    assert all(r['use_quality_only'].lower() == 'true' for r in split)

    counts = Counter(r['split'] for r in split)
    assert counts == {'train':2488,'val':552,'test':683}
    group_partitions = defaultdict(set)
    date_partitions = defaultdict(set)
    for row in split:
        group_partitions[row['split_group']].add(row['split'])
        date_partitions[row['conservative_group']].add(row['split'])
    assert len(group_partitions)==202 and all(len(v)==1 for v in group_partitions.values())
    assert len(date_partitions)==14 and all(len(v)>1 for v in date_partitions.values())

    train = [r for r in split if r['split']=='train']
    train_keys=ids(train)
    reduced_keys=ids(cnn_train)
    assert len(cnn_train)==1798 and reduced_keys <= train_keys
    assert {r['split_group'] for r in cnn_train}=={r['split_group'] for r in train}
    assert sum(r['use_quality_and_cnn'].lower()=='false' for r in cnn_train)==308
    embedded={(r['class'],r['filename']) for r in emb_manifest}
    assert len(emb_manifest)==3540 and len(embedded)==3540
    missing=train_keys-embedded
    assert len(missing)==52 and missing <= reduced_keys

    old_counts=Counter(r['decision'] for r in old_cnn)
    assert len(old_cnn)==3540 and old_counts == {'keep':2507,'suggest_skip':1033}
    assert len(neighbors)==3507
    hist=[]
    for low,high in [(-1,4),(4,8),(8,16),(16,24),(24,32),(32,64)]:
        hist.append(sum(low<int(r['pHash_distance'])<=high for r in neighbors))
    assert hist==[1389,900,953,225,39,1]

    # Independently reconstruct accuracy and unweighted macro-F1 from per-image predictions.
    val={}
    for experiment,correct,f1 in [('quality',533,0.9573913689481396),
                                   ('cnn_diversity',518,0.9181743494341305)]:
        scores=classification(read_rows(RESULTS/'stage_a'/f'{experiment}_predictions.csv'))
        assert scores['n']==552 and scores['correct']==correct
        assert math.isclose(scores['macro_f1'],f1,abs_tol=1e-12)
        val[experiment]=scores
    test=classification(read_rows(RESULTS/'stage_a'/'test_predictions.csv'))
    assert test['n']==683 and test['correct']==667
    assert math.isclose(test['macro_f1'],0.9159206350451213,abs_tol=1e-12)

    report={'original_unique':3879,'quality_exclusion_candidates':156,
            'quality_survivors':3723,'train_quality':2488,'train_cnn_diversity':1798,
            'val':552,'test':683,'V3_shooting_groups':202,
            'V3_shooting_groups_crossing_splits':0,
            'V3_class_date_groups_crossing_splits':14,
            'legacy_CNN_Balanced_suggest_skip':1033,
            'CNN_Diversity_old_CNN_flag_false':308,
            'Train_images_without_embedded_features_and_kept':52,
            'neighbor_pHash_hamming_buckets':hist,
            'stage_b_includes_all_quality_images':True,
            'stage_a_validation':val,'stage_a_test':test,
            'disclaimer':'Source image bytes and independent capture dates not checked by this script.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('\nALL FROZEN-CSV AND PREDICTION CHECKS PASSED')


if __name__=='__main__':
    main()
