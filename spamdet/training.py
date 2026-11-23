"""Offline model comparison; no inference-time fitting or downloads."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from threadpoolctl import threadpool_limits
from spamdet.curation import ROOT, SEED, canonical_text


def original_clean(text):
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', str(text).lower())
    text = re.sub(r'\d+', ' ', text)
    return re.sub(r'\s+', ' ', text.translate(str.maketrans('', '', string.punctuation))).strip()


def make_pipeline(name):
    word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000, sublinear_tf=name!='baseline_nb')
    if name == 'baseline_nb':
        word.set_params(preprocessor=original_clean)
        features, classifier = word, MultinomialNB()
    else:
        features = word if name == 'word_lr' else FeatureUnion([
            ('word',word),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5), min_df=2,max_features=20000,sublinear_tf=True))],n_jobs=1)
        classifier = LogisticRegression(C=4,solver='liblinear',max_iter=1000,random_state=SEED)
    return Pipeline([('features',features),('classifier',classifier)])


def spam_scores(pipeline, texts):
    return pipeline.predict_proba(texts)[:,list(pipeline.classes_).index('spam')]


def metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    cm = confusion_matrix(y,predicted,labels=['ham','spam'])
    tn,fp,fn,tp = (int(v) for v in cm.ravel())
    return dict(precision=float(precision_score(y,predicted,pos_label='spam',zero_division=0)),
                recall=float(recall_score(y,predicted,pos_label='spam',zero_division=0)),
                f1=float(f1_score(y,predicted,pos_label='spam',zero_division=0)),
                f0_5=float(fbeta_score(y,predicted,beta=.5,pos_label='spam',zero_division=0)),
                confusion_matrix=cm.tolist(),confusion_order=['ham','spam'],
                support={'ham':int(sum(y=='ham')),'spam':int(sum(y=='spam'))},
                false_positives=fp,false_positive_rate=float(fp/(tn+fp)) if tn+fp else 0.0)


def select_threshold(y,scores):
    # Enumerating all observed validation scores includes every realizable decision.
    thresholds = sorted(set(float(x) for x in scores)) + [float(np.nextafter(max(scores), np.inf))]
    choices = []
    for threshold in thresholds:
        result = metrics(y,np.where(scores>=threshold,'spam','ham'))
        if result['false_positive_rate'] <= .01:
            choices.append((result['f0_5'],result['recall'],-result['false_positive_rate'],threshold,result))
    best = max(choices,key=lambda x:x[:4])
    return best[3],best[4]


def train():
    rows = [json.loads(x) for x in (ROOT/'data/processed/curated.jsonl').read_text().splitlines()]
    curation = json.loads((ROOT/'reports/curation.json').read_text())
    report = dict(schema_version=1, seed=SEED, sklearn_version=sklearn.__version__,
                  split_sha256=curation['split_sha256'],score_semantics='Uncalibrated classifier scores, NOT calibrated probabilities.',
                  selection='Maximize validation F0.5 subject to empirical validation ham FPR <=1%; tie-break recall, lower FPR, higher threshold. Select model only using validation. Threshold fixed before test; no refit on validation/test.',
                  baseline='Algorithmic reproduction only: original lowercase URL/digit/punctuation removal + word TFIDF(1,2), min_df=2 + MultinomialNB on SAME SMS split. Feature cap 20000 is resource adaptation. Not reproduction of original app accuracy; standalone original failed cv2 import in parent verification.',
                  experiments={},selected={}, by_channel_language={},
                  indonesian={'status':'unvalidated','support':0,'reason':'No eligible licensed real Indonesian test set.'},
                  independent_source_test=curation['independent_source_test'],
                  combined_vs_per_channel='Not performed: only SMS has eligible accessible data; comparing combined/per-channel would be identical, not evidence of transfer.',
                  limitations=['Historical single-source SMS data; source-level English designation includes informal/mixed varieties.',
                               'No eligible accessible email corpus; official request blocked by approval layer and not retried.',
                               'Grouping is heuristic, not proof all semantic template leakage is eliminated.',
                               'Empirical validation FPR constraint is not a production guarantee; no calibration or prospective evaluation.',
                               'No cross-source/temporal/Indonesian validation.'],
                  artifact_contract_status='BLOCKED: required email pipeline cannot be honestly trained. Artifact intentionally contains only a genuine SMS entry; consumers must handle absent email. Never relabel SMS as email.')
    models = {}
    with threadpool_limits(limits=1):
        for channel in sorted({r['channel'] for r in rows}):
            subsets = {s:[r for r in rows if r['channel']==channel and r['split']==s] for s in ['train','val','test']}
            texts = {s:[canonical_text(r) for r in batch] for s,batch in subsets.items()}
            labels = {s:np.array([r['label'] for r in batch]) for s,batch in subsets.items()}
            candidates = {}
            names = ['baseline_nb','word_lr','word_char_lr'] if channel=='sms' else ['word_lr','word_char_lr']
            for name in names:
                pipeline = make_pipeline(name)
                pipeline.fit(texts['train'],labels['train'])
                threshold, validation = select_threshold(labels['val'],spam_scores(pipeline,texts['val']))
                candidates[name] = (pipeline,threshold,validation)
                report['experiments'][channel+'_'+name] = {'validation':validation,'threshold':threshold,'train_support':dict(Counter(labels['train'])),
                    'feature_count':len(pipeline.named_steps['features'].get_feature_names_out())}
            winner = max(names,key=lambda name:(candidates[name][2]['f0_5'],candidates[name][2]['recall'],-candidates[name][2]['false_positive_rate'],name))
            pipeline,threshold,validation = candidates[winner]
            # Test outputs are computed only after locking selection above.
            for name,(p,t,_) in candidates.items():
                pred = np.where(spam_scores(p,texts['test'])>=t,'spam','ham')
                report['experiments'][channel+'_'+name]['test'] = metrics(labels['test'],pred)
                if name=='baseline_nb':
                    report['experiments'][channel+'_'+name]['test_default_0_5'] = metrics(labels['test'],p.predict(texts['test']))
            for lang in sorted({r['language'] for r in subsets['test']}):
                batch = [r for r in subsets['test'] if r['language']==lang]
                scores = spam_scores(pipeline,[canonical_text(r) for r in batch])
                predicted = np.where(scores>=threshold,'spam','ham')
                result = metrics([r['label'] for r in batch],predicted)
                result['false_positive_row_ids'] = [r['row_id'] for r,p in zip(batch,predicted) if r['label']=='ham' and p=='spam']
                report['by_channel_language'][channel+'_'+lang] = result
            models[channel] = dict(pipeline=pipeline,threshold=float(threshold),name=winner)
            report['selected'][channel] = {'name':winner,'threshold':float(threshold)}
    # Exact entry shape for real models; deliberately no fake email placeholder.
    artifact = dict(models=models,metadata={'schema_version':1,'seed':SEED,'languages':['en'],
                    'unavailable_channels':['email'] if 'email' not in models else [],
                    'score_semantics':report['score_semantics'],'split_sha256':report['split_sha256']})
    (ROOT/'artifacts').mkdir(exist_ok=True)
    joblib.dump(artifact,ROOT/'artifacts/model.joblib')
    restored = joblib.load(ROOT/'artifacts/model.joblib')
    assert set(restored) == {'models','metadata'}
    for entry in restored['models'].values():
        assert set(entry) == {'pipeline','threshold','name'}
        assert list(entry['pipeline'].classes_) == ['ham','spam']
        assert isinstance(entry['threshold'],float)
    stable = json.dumps(report,sort_keys=True)
    report['metrics_sha256'] = hashlib.sha256(stable.encode()).hexdigest()
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report
