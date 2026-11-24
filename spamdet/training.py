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
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from threadpoolctl import threadpool_limits
from spamdet.curation import ROOT, SEED, canonical_text, read_jsonl


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
        classifier = (CalibratedClassifierCV(LinearSVC(C=1, random_state=SEED), cv=3)
                      if name == 'word_char_svm' else
                      LogisticRegression(C=4,solver='liblinear',max_iter=1000,random_state=SEED))
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
                f2=float(fbeta_score(y,predicted,beta=2,pos_label='spam',zero_division=0)),
                f0_5=float(fbeta_score(y,predicted,beta=.5,pos_label='spam',zero_division=0)),
                confusion_matrix=cm.tolist(),confusion_order=['ham','spam'],
                support={'ham':int(sum(y=='ham')),'spam':int(sum(y=='spam'))},
                false_positives=fp,false_negatives=fn,false_positive_rate=float(fp/(tn+fp)) if tn+fp else 0.0)


def select_threshold(y,scores):
    # Policy declared before training: maximize validation F2, ham FPR <=3%.
    # Enumerate tied score blocks once; no manual diagnostic/test-set inputs.
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if not len(y) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Validation scores must be finite in [0,1].')
    if set(y) != {'ham','spam'}:
        raise ValueError('Validation requires both ham and spam.')
    order = np.argsort(-scores, kind='stable')
    sorted_scores, positive = scores[order], y[order] == 'spam'
    tp, fp = np.cumsum(positive), np.cumsum(~positive)
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y)-1]
    positives, negatives = int(positive.sum()), int((~positive).sum())
    choices = []
    if sorted_scores[0] < 1:
        choices.append((0., 0., 0., float(np.nextafter(sorted_scores[0], np.inf))))
    for i in ends:
        fpr = float(fp[i] / negatives)
        if fpr <= .03:
            f2 = float(5*tp[i] / (5*tp[i] + 4*(positives-tp[i]) + fp[i]))
            choices.append((f2, float(tp[i]/positives), -fpr, float(sorted_scores[i])))
    if not choices:
        raise ValueError('No threshold in [0,1] satisfies validation ham FPR <=3%.')
    threshold = max(choices)[3]
    return threshold, metrics(y, np.where(scores >= threshold, 'spam', 'ham'))


def train():
    rows = read_jsonl(ROOT/'data/processed/curated.jsonl')
    curation = json.loads((ROOT/'reports/curation.json').read_text(encoding='utf-8'))
    report = dict(schema_version=2, seed=SEED, sklearn_version=sklearn.__version__,
                  split_sha256=curation['split_sha256'], score_semantics='Classifier scores in [0,1], not validated confidence. SVM sigmoid calibration uses training CV only.',
                  selection='Maximize validation F2 subject to empirical validation ham FPR <=3%; tie-break recall, lower FPR, higher threshold. Model selected on validation only; no refit on validation/test.',
                  experiments={}, selected={}, by_channel_language={},
                  independent_source_test=curation['independent_source_test'],
                  limitations=['Email Indonesian unvalidated: no eligible original holdout.',
                               'Historical and topic-specific corpora; no prospective/independent-source evaluation.',
                               'Grouped splitting is heuristic, not proof of zero semantic leakage.',
                               'Promo labels do not encode recipient consent; OTP/invoice/refund coverage not established.',
                               'Validation FPR cap is not a production guarantee.'])
    models = {}
    with threadpool_limits(limits=1):
        for channel in sorted({r['channel'] for r in rows}):
            subsets = {s:[r for r in rows if r['channel']==channel and r['split']==s] for s in ['train','val','test']}
            texts = {s:[canonical_text(r) for r in batch] for s,batch in subsets.items()}
            labels = {s:np.array([r['label'] for r in batch]) for s,batch in subsets.items()}
            candidates = {}
            names = ['baseline_nb','word_lr','word_char_lr','word_char_svm']
            for name in names:
                print('Training ' + channel + '/' + name, flush=True)
                pipeline = make_pipeline(name)
                pipeline.fit(texts['train'],labels['train'])
                threshold, validation = select_threshold(labels['val'],spam_scores(pipeline,texts['val']))
                candidates[name] = (pipeline,threshold,validation)
                report['experiments'][channel+'_'+name] = {'validation':validation,'threshold':threshold,
                    'train_support':dict(Counter(labels['train'])),
                    'feature_count':len(pipeline.named_steps['features'].get_feature_names_out())}
            winner = max(names,key=lambda name:(candidates[name][2]['f2'],candidates[name][2]['recall'],-candidates[name][2]['false_positive_rate'],name))
            pipeline,threshold,_ = candidates[winner]
            # Selection is locked above before ANY test predictions are inspected.
            for name,(p,t,_) in candidates.items():
                experiment = report['experiments'][channel+'_'+name]
                experiment['test'] = metrics(labels['test'],np.where(spam_scores(p,texts['test'])>=t,'spam','ham'))
                experiment['by_language'] = {}
                for lang in sorted({r['language'] for r in subsets['test']}):
                    batch = [r for r in subsets['test'] if r['language']==lang]
                    prediction = np.where(spam_scores(p,[canonical_text(r) for r in batch])>=t,'spam','ham')
                    result = metrics([r['label'] for r in batch],prediction)
                    experiment['by_language'][lang] = result
                    if name == winner:
                        report['by_channel_language'][channel+'_'+lang] = dict(result,
                            false_positive_row_ids=[r['row_id'] for r,v in zip(batch,prediction) if r['label']=='ham' and v=='spam'],
                            false_negative_row_ids=[r['row_id'] for r,v in zip(batch,prediction) if r['label']=='spam' and v=='ham'])
            models[channel] = dict(pipeline=pipeline,threshold=float(threshold),name=winner)
            report['selected'][channel] = {'name':winner,'threshold':float(threshold)}

        baseline_dir = ROOT/'artifacts/baseline'
        if (baseline_dir/'model.joblib').is_file() and (baseline_dir/'curated.jsonl').is_file():
            from spamdet.curation import normalized
            old_rows = read_jsonl(baseline_dir/'curated.jsonl')
            seen = {normalized(canonical_text(r)) for r in old_rows if r['split'] != 'test'}
            contaminated = {r['group'] for r in rows if normalized(canonical_text(r)) in seen}
            safe = [r for r in rows if r['split']=='test' and r['group'] not in contaminated]
            old = joblib.load(baseline_dir/'model.joblib')
            comparison = {'scope':'New test rows excluding every new group touching old train or validation text. Same rows for old/new; old has no email pipeline.', 'by_channel_language':{}}
            for channel in old['models']:
                for lang in sorted({r['language'] for r in safe if r['channel']==channel}):
                    batch = [r for r in safe if r['channel']==channel and r['language']==lang]
                    entry = {}
                    for version, model in [('old', old['models'][channel]), ('new', models[channel])]:
                        prediction = np.where(spam_scores(model['pipeline'],[canonical_text(r) for r in batch])>=model['threshold'],'spam','ham')
                        entry[version] = metrics([r['label'] for r in batch], prediction)
                    comparison['by_channel_language'][channel+'_'+lang] = entry
            report['old_new_common_test'] = comparison

    metadata = {'schema_version':2,'seed':SEED,'languages':sorted({r['language'] for r in rows}),
                'languages_by_channel':{c:sorted({r['language'] for r in rows if r['channel']==c and r['split']=='test'}) for c in models},
                'unavailable_channels':sorted({'sms','email'}-set(models)),
                'score_semantics':report['score_semantics'],'split_sha256':report['split_sha256'],
                'training_counts':curation['splits']['train'], 'sources':{k:v['sha256'] for k,v in curation['sources'].items()},
                'selection':report['selection'], 'sklearn_version':sklearn.__version__}
    artifact = dict(models=models,metadata=metadata)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    temporary = ROOT/'artifacts/model.pending.joblib'
    joblib.dump(artifact, temporary)
    restored = joblib.load(temporary)
    for channel, entry in restored['models'].items():
        assert list(entry['pipeline'].classes_) == ['ham','spam']
        assert 0 <= entry['threshold'] <= 1
        assert np.isfinite(spam_scores(entry['pipeline'], ['validation load probe'])).all()
    temporary.replace(ROOT/'artifacts/model.joblib')
    report['artifact_sha256'] = hashlib.sha256((ROOT/'artifacts/model.joblib').read_bytes()).hexdigest()
    report['metrics_sha256'] = hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'selected':report['selected'],'by_channel_language':report['by_channel_language']},indent=2))
    return report
