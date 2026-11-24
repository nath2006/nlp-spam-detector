"""Precision refinement with immutable external holdout and historical scope labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from spamdet.curation import ROOT, canonical_text, curate, digest, normalized, read_jsonl
from spamdet.training import make_pipeline, metrics, spam_scores

HISTORY = ROOT/'artifacts/history/precision_round'
FROZEN = ROOT/'reports/frozen_precision_holdout.json'
ROWS = ROOT/'data/processed/precision_rows.jsonl'
POLICY = 'One threshold/channel; maximize macro validation F2, ham FPR <=3% in each group with >=100 ham and >=30 spam; no inferred-language routing.'


def select_group_threshold(labels, scores, groups):
    labels, scores, groups = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(groups)
    if labels.ndim != 1 or scores.shape != labels.shape or groups.shape != labels.shape or set(labels) != {'ham','spam'}:
        raise ValueError('Require aligned binary validation labels, scores and groups.')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be finite in [0,1].')
    thresholds = np.unique(scores)
    if thresholds[-1] < 1:
        thresholds = np.r_[thresholds, np.nextafter(thresholds[-1], np.inf)]
    masks = {str(g): groups == g for g in np.unique(groups)}
    choices = []
    for threshold in thresholds:
        prediction = scores >= threshold
        f2_values, recalls, fprs = [], [], []
        feasible = True
        for mask in masks.values():
            positive, negative = (labels[mask] == 'spam'), (labels[mask] == 'ham')
            tp = int(np.sum(prediction[mask] & positive))
            fp = int(np.sum(prediction[mask] & negative))
            nspam, nham = int(positive.sum()), int(negative.sum())
            fpr = fp/nham if nham else 0.
            if nham >= 100 and nspam >= 30 and fpr > .03:
                feasible = False
                break
            if nspam:
                f2_values.append(5*tp/(5*tp+4*(nspam-tp)+fp))
                recalls.append(tp/nspam)
            if nham:
                fprs.append(fpr)
        if feasible:
            choices.append((float(np.mean(f2_values)),float(np.mean(recalls)),-max(fprs,default=0.),float(threshold)))
    if not choices:
        raise ValueError('No feasible bounded threshold for per-group FPR policy.')
    threshold = max(choices)[3]
    report = group_metrics(labels, scores, groups, threshold)
    return threshold, report


def group_metrics(labels, scores, groups, threshold):
    labels, scores, groups = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    result = {}
    for group in sorted(set(groups)):
        mask = groups == group
        measured = metrics(labels[mask], np.where(scores[mask] >= threshold, 'spam', 'ham'))
        ham, spam = measured['support']['ham'], measured['support']['spam']
        measured['constraint_eligible'] = ham >= 100 and spam >= 30
        measured['target_met_empirical'] = measured['false_positive_rate'] <= .03 if measured['constraint_eligible'] else None
        if not ham:
            measured['false_positive_rate'] = None
        result[str(group)] = measured
    return result


def allocate_groups(rows):
    batches = defaultdict(list)
    for row in rows:
        batches[row['group']].append(row)
    output = []
    for group, batch in batches.items():
        old_splits = {r['origin_split'] for r in batch if r.get('origin_split')}
        if len(old_splits) > 1:
            assigned = [(r, 'quarantine') for r in batch]
        elif old_splits:
            original = next(iter(old_splits))
            assigned = []
            for row in batch:
                if row.get('origin_split'):
                    split = 'historical_test' if original == 'test' else original
                elif row['source_id'] == 'india_sms':
                    split = 'excluded_overlap'
                else:
                    split = 'train' if original == 'train' else 'excluded_holdout'
                assigned.append((row, split))
        else:
            has_external = any(r['source_id'] == 'india_sms' for r in batch)
            fold = int(digest('precision-20261124:' + group)[:12], 16) % 5
            external_split = 'new_test' if fold < 2 else 'val' if fold == 2 else 'train'
            assigned = [(r, external_split if r['source_id'] == 'india_sms' else
                         'excluded_holdout' if has_external and external_split != 'train' else 'train') for r in batch]
        for row, split in assigned:
            output.append(dict(row, split=split, train_only=bool(row.get('original_train_only', row.get('synthetic',False)))))
    return sorted(output, key=lambda row: row['row_id'])


def snapshot():
    if (HISTORY/'snapshot.json').exists():
        expected = json.loads((HISTORY/'snapshot.json').read_text(encoding='utf-8'))
        for filename, value in expected.items():
            if hashlib.sha256((HISTORY/filename).read_bytes()).hexdigest() != value:
                raise ValueError('Historical snapshot changed: ' + filename)
        return
    HISTORY.mkdir(parents=True, exist_ok=True)
    files = {'model.joblib':'artifacts/model.joblib','evaluation.json':'reports/evaluation.json',
             'curation.json':'reports/curation.json','curated.jsonl':'data/processed/curated.jsonl',
             'sources.json':'data/sources.json'}
    for filename, path in files.items():
        shutil.copy2(ROOT/path, HISTORY/filename)
    (HISTORY/'snapshot.json').write_text(json.dumps({name:hashlib.sha256((HISTORY/name).read_bytes()).hexdigest() for name in files},indent=2),encoding='utf-8')


def load_additional():
    specs = json.loads((ROOT/'data/precision_sources.json').read_text(encoding='utf-8'))
    rows, audit = [], {}
    for source in specs['sources']:
        path = ROOT/source['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Source hash mismatch: ' + source['id'])
        original = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
        counts = Counter()
        selected = []
        for index, raw in enumerate(original, 1):
            body = raw['Msg'] if source['format'] == 'india_csv' else raw['text']
            label = raw['Label'] if source['format'] == 'india_csv' else raw['label']
            counts[label] += 1
            if label not in ('ham','spam','phishing'):
                raise ValueError('Unmapped additional-source label')
            language = 'en' if source['format'] == 'india_csv' else raw['language']
            if language not in ('en','id') or body.strip().lower() in ('?image omitted','image omitted','<media omitted>') or not body.strip():
                continue
            selected.append(dict(channel='sms',language=language,label='ham' if label=='ham' else 'spam',subject='',body=body.strip(),
                                 row_id=source['id']+':'+str(index),source_id=source['id'],thread_ids=[],
                                 synthetic=source.get('synthetic',False),original_train_only=source.get('train_only',False),
                                 category=raw.get('category','source_binary'),original_label=label))
        if source['format'] == 'ots_csv':
            buckets = defaultdict(list)
            for row in selected:
                buckets[(row['language'],row['label'],row['category'])].append(row)
            selected = []
            for (language,_,_), batch in sorted(buckets.items()):
                batch.sort(key=lambda r: digest(r['body']))
                selected.extend(batch[:source['max_en_per_category']] if language=='en' else batch)
        rows.extend(selected)
        audit[source['id']] = dict(raw_count=len(original),raw_label_counts=dict(counts),eligible_before_grouping=len(selected),
                                  selected_label_counts=dict(Counter(r['label'] for r in selected)),
                                  selected_language_counts=dict(Counter(r['language'] for r in selected)),license=source['license'],sha256=source['sha256'])
    return rows, audit


def freeze():
    snapshot()
    if FROZEN.exists():
        frozen = json.loads(FROZEN.read_text(encoding='utf-8'))
        if hashlib.sha256(ROWS.read_bytes()).hexdigest() != frozen['dataset_sha256']:
            raise ValueError('Frozen precision rows changed; refusing resplit.')
        if hashlib.sha256((ROOT/'data/precision_sources.json').read_bytes()).hexdigest() != frozen['source_manifest_sha256']:
            raise ValueError('Frozen source manifest changed; use a separately planned round.')
        return frozen
    old = read_jsonl(HISTORY/'curated.jsonl')
    for row in old:
        row['origin_split'] = row['split']
        row['original_train_only'] = row.get('train_only',False)
        row['thread_ids'] = row.get('thread_ids',[]) + ['snapshot-group:' + row['group']]
    additional, source_audit = load_additional()
    combined, audit = curate(old + additional)
    allocated = allocate_groups(combined)
    active_splits = defaultdict(set)
    for row in allocated:
        if row['split'] in ('train','val','historical_test','new_test'):
            active_splits[row['group']].add(row['split'])
        assert not row.get('synthetic') or row['split'] in ('train','excluded_holdout','quarantine')
    assert all(len(values)==1 for values in active_splits.values())
    manual = json.loads((ROOT/'examples/diagnostic_messages.json').read_text(encoding='utf-8'))
    forbidden = {normalized(canonical_text(dict(r,subject=r.get('subject','')))) for r in manual}
    assert not any(normalized(canonical_text(r)) in forbidden for r in allocated)
    ROWS.parent.mkdir(exist_ok=True,parents=True)
    ROWS.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allocated),encoding='utf-8')
    record = dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,
                  baseline_sha256=hashlib.sha256((HISTORY/'model.joblib').read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256((ROOT/'data/precision_sources.json').read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(ROWS.read_bytes()).hexdigest(),group_audit=audit,source_audit=source_audit,
                  split_counts={s:dict(Counter(r['channel']+'_'+r['language']+'_'+r['label'] for r in allocated if r['split']==s)) for s in sorted({r['split'] for r in allocated})},
                  new_test=[{'row_id':r['row_id'],'group':r['group'],'text_hash':r['text_hash']} for r in allocated if r['split']=='new_test'],
                  scope='New external original SMS English/Hinglish only. SMS-ID/email-EN are historical reuse, not new independent holdouts. Email-ID absent.')
    FROZEN.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return record


def train_candidates():
    frozen = freeze()
    rows = read_jsonl(ROWS)
    baseline = joblib.load(HISTORY/'model.joblib')
    candidates, selected, validation = {}, {}, {}
    with threadpool_limits(limits=1):
        for channel in ('sms','email'):
            val = [r for r in rows if r['channel']==channel and r['split']=='val']
            train = [r for r in rows if r['channel']==channel and r['split']=='train']
            labels = np.array([r['label'] for r in val])
            groups = np.array([channel+'_'+r['language'] for r in val])
            texts = [canonical_text(r) for r in val]
            current = baseline['models'][channel]
            names = ['current_retuned'] + (['word_char_lr_aug','word_char_svm_aug'] if channel=='sms' else [])
            for name in names:
                print('Validation candidate '+channel+'/'+name, flush=True)
                if name == 'current_retuned':
                    pipeline = current['pipeline']
                else:
                    pipeline = make_pipeline(name.removesuffix('_aug'))
                    pipeline.fit([canonical_text(r) for r in train],[r['label'] for r in train])
                scores = spam_scores(pipeline,texts)
                threshold, measured = select_group_threshold(labels,scores,groups)
                key = channel+'/'+name
                candidates[key] = dict(pipeline=pipeline,threshold=threshold,name=name)
                validation[key] = dict(threshold=threshold,by_group=measured,
                                       macro_f2=float(np.mean([m['f2'] for m in measured.values()])),
                                       macro_recall=float(np.mean([m['recall'] for m in measured.values()])))
            baseline_measured = group_metrics(labels,spam_scores(current['pipeline'],texts),groups,current['threshold'])
            validation[channel+'/baseline_fixed'] = dict(threshold=current['threshold'],by_group=baseline_measured,
                macro_f2=float(np.mean([m['f2'] for m in baseline_measured.values()])),
                macro_recall=float(np.mean([m['recall'] for m in baseline_measured.values()])))
            keys = [channel+'/'+name for name in names]
            selected[channel] = max(keys,key=lambda key:(validation[key]['macro_f2'],validation[key]['macro_recall'],key))
    record = dict(policy=POLICY,selected=selected,candidates=validation,
                  frozen_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                  created_utc=datetime.now(timezone.utc).isoformat(),test_read_for_selection=False)
    joblib.dump(candidates, ROOT/'artifacts/precision_candidates.joblib')
    record['candidates_sha256'] = hashlib.sha256((ROOT/'artifacts/precision_candidates.joblib').read_bytes()).hexdigest()
    (ROOT/'reports/precision_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return record


def evaluate_and_install():
    frozen = freeze()
    lock = json.loads((ROOT/'reports/precision_validation.json').read_text(encoding='utf-8'))
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != lock['frozen_sha256']:
        raise ValueError('Frozen holdout changed since validation.')
    candidate_path = ROOT/'artifacts/precision_candidates.joblib'
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != lock['candidates_sha256']:
        raise ValueError('Candidate artifact changed after selection.')
    candidates = joblib.load(candidate_path)
    baseline = joblib.load(HISTORY/'model.joblib')
    rows = read_jsonl(ROWS)
    report = dict(schema_version=3,selection=POLICY,selected={},by_channel_language={},
                  validation=lock,new_holdout={},historical_comparison={},
                  frozen_sha256=lock['frozen_sha256'],split_sha256=frozen['dataset_sha256'],
                  evaluation_scope='Historical grouped test reuse for by_channel_language; new external SMS English/Hinglish is separate. No original new SMS-ID/email-EN/email-ID holdout.',
                  new_holdout_scope=frozen['scope'],limitations=['Empirical per-group validation cap is not a population/production guarantee.',
                  'Historical SMS-ID and email-EN tests were inspected in earlier development; not fresh blind evaluation.',
                  'Transactional augmentation is synthetic SMS, not real email or original Indonesian test data.',
                  'Consent of individual source promotional messages is not annotated.'],source_audit=frozen['source_audit'])
    models = {channel:candidates[key] for channel,key in lock['selected'].items()}
    with threadpool_limits(limits=1):
        for channel in ('sms','email'):
            entries = {'baseline_fixed':baseline['models'][channel]}
            entries.update({key.split('/')[1]:value for key,value in candidates.items() if key.startswith(channel+'/')})
            for scope,split in [('new_holdout','new_test'),('historical_comparison','historical_test')]:
                batch = [r for r in rows if r['split']==split and r['channel']==channel]
                if not batch:
                    continue
                labels = [r['label'] for r in batch]
                groups = [channel+'_'+r['language'] for r in batch]
                texts = [canonical_text(r) for r in batch]
                for name,entry in entries.items():
                    measured = group_metrics(labels,spam_scores(entry['pipeline'],texts),groups,entry['threshold'])
                    report[scope][channel+'/'+name] = measured
                    if scope=='historical_comparison' and name==models[channel]['name']:
                        for group,result in measured.items():
                            report['by_channel_language'][group] = dict(result,evaluation_scope='historical_reuse')
            report['selected'][channel] = dict(name=models[channel]['name'],threshold=models[channel]['threshold'])
    metadata = dict(baseline['metadata'])
    actual_counts = {}
    for channel,entry in models.items():
        if entry['name'].endswith('_aug'):
            actual_counts.update({k:v for k,v in frozen['split_counts']['train'].items() if k.startswith(channel+'_')})
        else:
            actual_counts.update({k:v for k,v in baseline['metadata']['training_counts'].items() if k.startswith(channel+'_')})
    metadata.update(schema_version=3,selection=POLICY,split_sha256=frozen['dataset_sha256'],training_counts=actual_counts,
                    baseline_sha256=frozen['baseline_sha256'],frozen_sha256=lock['frozen_sha256'],
                    refinement_sources={k:v['sha256'] for k,v in frozen['source_audit'].items()},
                    threshold_routing='one per channel; no language detector',indonesian_email_validated=False)
    temporary = ROOT/'artifacts/model.pending.joblib'
    joblib.dump(dict(models=models,metadata=metadata),temporary)
    restored = joblib.load(temporary)
    for channel,entry in restored['models'].items():
        assert set(entry['pipeline'].classes_) == {'ham','spam'}
        assert 0 <= entry['threshold'] <= 1
        assert np.isfinite(spam_scores(entry['pipeline'],['artifact verification'])).all()
    temporary.replace(ROOT/'artifacts/model.joblib')
    report['artifact_sha256'] = hashlib.sha256((ROOT/'artifacts/model.joblib').read_bytes()).hexdigest()
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (ROOT/'reports/precision_evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
