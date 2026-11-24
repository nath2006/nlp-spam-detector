"""Bounded SMS-only experiment. Never trains/selects using diagnostic examples.

Stages: train freezes policy/snapshot and selects on validation; evaluate reports
all fixed candidates on existing tests but cannot change the selection.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from spamdet.curation import ROOT,canonical_text,read_jsonl
from spamdet.refinement import freeze,group_metrics,select_group_threshold
from spamdet.sms_recall import training_weights,promotion_gate,sms_only_bundle
from spamdet.training import make_pipeline,spam_scores

OUT=ROOT/'reports/sms_recall'
ART=ROOT/'artifacts/sms_recall'
SNAP=ART/'before'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def load(path): return json.loads(path.read_text(encoding='utf-8'))

def train():
    frozen=freeze()
    OUT.mkdir(exist_ok=True,parents=True); ART.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan.json').exists(): raise ValueError('Round already planned: refuse to silently rerun search.')
    SNAP.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'artifacts/model.joblib',SNAP/'model.joblib')
    shutil.copy2(ROOT/'reports/evaluation.json',SNAP/'evaluation.json')
    protected=['app.py','spamdet/inference.py','tests/test_real_artifact.py','examples/diagnostic_messages.json']
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),active_sha256=sha(SNAP/'model.joblib'),
              frozen_sha256=sha(ROOT/'reports/frozen_precision_holdout.json'),dataset_sha256=frozen['dataset_sha256'],
              protected_sha256={p:sha(ROOT/p) for p in protected},
              objective='One SMS threshold; per-language validation ham FPR <=3% for >=100 ham and >=30 spam. Threshold maximizes macro F2 (existing selector). Candidate ranking: macro recall, minimum language recall, macro F2, fewer FP. No manual or test-based selection.',
              promotion_policy='Selected candidate only: same support on all SMS test cohorts; FPR <=3% each, no additional FN on any cohort, strict FN reduction on at least one. No fallback to another candidate after tests. Manual examples only smoke after locked evaluation.',
              variants=[dict(C=c,synthetic_weight=s,balance_language_label=b,wide=False) for c in (1,4,16) for s in (.1,1.) for b in (False,True)]+
                       [dict(C=4,synthetic_weight=.1,balance_language_label=b,wide=True) for b in (False,True)],
              new_indonesian_holdout='No eligible new original licensed source verified; use frozen historical ID and label it reused. English external test also inspected in previous round, not fresh blind.',
              forbidden_inputs='No examples/diagnostic_messages.json read for fitting, threshold choice or candidate ranking. No email training or feature changes.')
    save(OUT/'plan.json',plan)
    active=joblib.load(SNAP/'model.joblib'); baseline=joblib.load(ROOT/'artifacts/history/precision_round/model.joblib')
    rows=[r for r in read_jsonl(ROOT/'data/processed/precision_rows.jsonl') if r['channel']=='sms']
    train_rows=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='val']
    x=[canonical_text(r) for r in train_rows]; y=[r['label'] for r in train_rows]
    vx=[canonical_text(r) for r in val]; vy=[r['label'] for r in val]; vg=['sms_'+r['language'] for r in val]
    entries={}; validation={}
    with threadpool_limits(limits=1):
        for index,spec in enumerate(plan['variants']):
            name='lr_'+str(index)
            print('FIT',name,spec,flush=True)
            pipe=make_pipeline('word_char_lr')
            pipe.set_params(classifier__C=spec['C'])
            if spec['wide']:
                pipe.set_params(features__word__min_df=1,features__char__min_df=1,
                                features__char__ngram_range=(2,5),features__char__max_features=40000)
            weights=training_weights(train_rows,spec['synthetic_weight'],spec['balance_language_label'])
            pipe.fit(x,y,classifier__sample_weight=weights)
            threshold,measured=select_group_threshold(vy,spam_scores(pipe,vx),vg)
            entries[name]=dict(pipeline=pipe,threshold=threshold,name='sms_recall_'+name)
            validation[name]=dict(spec=spec,threshold=threshold,by_group=measured)
        for name,entry in [('baseline_retuned',baseline['models']['sms']),('active_retuned',active['models']['sms'])]:
            threshold,measured=select_group_threshold(vy,spam_scores(entry['pipeline'],vx),vg)
            entries[name]=dict(pipeline=entry['pipeline'],threshold=threshold,name='sms_recall_'+name)
            validation[name]=dict(spec='pipeline unchanged; threshold validation-only',threshold=threshold,by_group=measured)
        for name,entry in [('baseline_fixed',baseline['models']['sms']),('active_fixed',active['models']['sms'])]:
            validation[name]=dict(threshold=entry['threshold'],by_group=group_metrics(vy,spam_scores(entry['pipeline'],vx),vg,entry['threshold']))
    def rank(name):
        ms=list(validation[name]['by_group'].values())
        return (float(np.mean([m['recall'] for m in ms])),min(m['recall'] for m in ms),
                float(np.mean([m['f2'] for m in ms])),-sum(m['false_positives'] for m in ms),name)
    winner=max(entries,key=rank)
    joblib.dump(entries,ART/'candidates.joblib')
    result=dict(selected=winner,selection_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),
                candidates_sha256=sha(ART/'candidates.joblib'),candidates=validation,rank=list(rank(winner)),
                test_used_for_selection=False,diagnostic_used_for_selection=False,
                training_counts=dict(Counter(r['language']+'_'+r['label'] for r in train_rows)))
    save(OUT/'validation.json',result)
    print('SELECTED',winner,result['rank'],flush=True)
    print(json.dumps({k:{g:{m:v[m] for m in ['support','recall','false_positives','false_negatives','false_positive_rate']} for g,v in r['by_group'].items()} for k,r in validation.items()},indent=2))

def evaluate():
    plan=load(OUT/'plan.json'); validation=load(OUT/'validation.json');freeze()
    assert validation['plan_sha256']==sha(OUT/'plan.json')
    assert validation['candidates_sha256']==sha(ART/'candidates.joblib')
    assert plan['active_sha256']==sha(ROOT/'artifacts/model.joblib')==sha(SNAP/'model.joblib')
    entries=joblib.load(ART/'candidates.joblib')
    active=joblib.load(SNAP/'model.joblib');baseline=joblib.load(ROOT/'artifacts/history/precision_round/model.joblib')
    entries.update(baseline_fixed=baseline['models']['sms'],active_fixed=active['models']['sms'])
    rows=read_jsonl(ROOT/'data/processed/precision_rows.jsonl');results={k:{} for k in entries}
    with threadpool_limits(limits=1):
        for split in ['historical_test','new_test']:
            batch=[r for r in rows if r['channel']=='sms' and r['split']==split]
            for name,entry in entries.items():
                groups=group_metrics([r['label'] for r in batch],spam_scores(entry['pipeline'],[canonical_text(r) for r in batch]),['sms_'+r['language'] for r in batch],entry['threshold'])
                results[name].update({split+'/'+g:v for g,v in groups.items()})
    winner=validation['selected'];gate=promotion_gate(results['active_fixed'],results[winner])
    report=dict(selected=winner,validation_sha256=sha(OUT/'validation.json'),results=results,gate=gate,
                evaluation_utc=datetime.now(timezone.utc).isoformat(),active_replaced=False,
                scope='All tests reused from previous round, not fresh independent validation; no eligible new ID holdout.',
                email_fingerprint_before=joblib.hash(active['models']['email']))
    save(OUT/'evaluation.json',report)
    print('LOCKED_SELECTED',winner,'GATE',gate)
    for name in ['baseline_fixed','active_fixed',winner]:
        print(name,json.dumps(results[name],indent=2))

def promote():
    plan=load(OUT/'plan.json'); validation=load(OUT/'validation.json'); report=load(OUT/'evaluation.json')
    assert report['validation_sha256']==sha(OUT/'validation.json')
    assert validation['plan_sha256']==sha(OUT/'plan.json')
    assert plan['frozen_sha256']==sha(ROOT/'reports/frozen_precision_holdout.json')
    assert validation['candidates_sha256']==sha(ART/'candidates.joblib')
    assert plan['active_sha256']==sha(ROOT/'artifacts/model.joblib')==sha(SNAP/'model.joblib')
    if not report['gate']['accepted']: raise ValueError('Selected SMS failed promotion gate; active retained.')
    entries=joblib.load(ART/'candidates.joblib');selected=report['selected']
    assert promotion_gate(report['results']['active_fixed'],report['results'][selected])['accepted']
    active=joblib.load(SNAP/'model.joblib')
    provenance=dict(validation_sha256=sha(OUT/'validation.json'),selected=selected,
                    active_before_sha256=plan['active_sha256'],policy=plan['objective'])
    bundle=sms_only_bundle(active,entries[selected],report['gate'],provenance)
    temporary=ART/'promoted.pending.joblib';joblib.dump(bundle,temporary)
    restored=joblib.load(temporary)
    assert joblib.hash(restored['models']['email'])==report['email_fingerprint_before']
    assert np.isfinite(spam_scores(restored['models']['sms']['pipeline'],['artifact verification'])).all()
    active_report=load(SNAP/'evaluation.json')
    active_report['sms_recall']=dict(selected=selected,results=report['results'][selected],
                                   active_before=report['results']['active_fixed'],scope=report['scope'])
    active_report['selected']['sms']=dict(name=entries[selected]['name'],threshold=entries[selected]['threshold'])
    for key,measured in report['results'][selected].items():
        split,group=key.split('/')
        if split=='historical_test':
            active_report['by_channel_language'][group]=dict(measured,evaluation_scope='historical_reuse')
    active_report['new_holdout']['sms/recall_active']={'sms_en':report['results'][selected]['new_test/sms_en']}
    active_report['historical_comparison']['sms/recall_active']={k.split('/')[1]:v for k,v in report['results'][selected].items() if k.startswith('historical_test/')}
    active_report['validation']['candidates']['sms/recall_active']=validation['candidates'][selected]
    active_report['validation']['selected']['sms']='sms/recall_active'
    active_report['validation']['policy']=plan['objective']+' Email selection unchanged.'
    active_report['selection']=plan['objective']+' Email selection unchanged.'
    active_report['artifact_sha256']=sha(temporary)
    # Preserve all email data exactly; change only SMS-specific entries and bundle hash.
    old_report=load(SNAP/'evaluation.json')
    assert active_report['by_channel_language']['email_en']==old_report['by_channel_language']['email_en']
    assert active_report['selected']['email']==old_report['selected']['email']
    temporary.replace(ROOT/'artifacts/model.joblib')
    save(ROOT/'reports/evaluation.json',active_report)
    report.update(active_replaced=True,artifact_sha256=sha(ROOT/'artifacts/model.joblib'),
                  email_fingerprint_after=joblib.hash(joblib.load(ROOT/'artifacts/model.joblib')['models']['email']),
                  promoted_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'evaluation.json',report)
    print('PROMOTED',selected,'HASH',report['artifact_sha256'],'EMAIL_UNCHANGED',report['email_fingerprint_before']==report['email_fingerprint_after'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','promote']);args=parser.parse_args()
    {'train':train,'evaluate':evaluate,'promote':promote}[args.stage]()
