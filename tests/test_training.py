import numpy as np
from spamdet.training import make_pipeline, select_threshold, metrics, original_clean


def test_features_fit_only_training_and_contract():
    p = make_pipeline('word_lr')
    p.fit(['ordinary meeting today','ordinary lunch today','win money prize','claim money prize'], ['ham','ham','spam','spam'])
    assert list(p.classes_) == ['ham','spam']
    assert 'unseenholdout' not in p.named_steps['features'].get_feature_names_out()
    assert set(p.named_steps) == {'features','classifier'}


def test_threshold_respects_false_positive_constraint():
    y = np.array(['ham']*100+['spam']*3)
    scores = np.array([0.1]*99+[0.8,0.7,0.8,0.9])
    t, m = select_threshold(y,scores)
    assert m['false_positive_rate'] <= .01
    assert metrics(y, np.where(scores>=t,'spam','ham')) == m


def test_baseline_clean():
    assert original_clean('HI! 123 https://example.org/test') == 'hi'


def test_metrics_reports_false_negatives_and_recall_weighted_f2():
    result = metrics(['ham','ham','spam','spam'], ['ham','spam','ham','spam'])
    assert result['false_positives'] == 1 and result['false_negatives'] == 1
    assert result['f2'] == .5


def test_threshold_uses_predeclared_recall_policy_not_manual_examples():
    y = np.array(['ham']*100 + ['spam']*11)
    scores = np.array([.01]*97 + [.7]*3 + [.6]*10 + [.9])
    threshold, result = select_threshold(y, scores)
    assert threshold == .6
    assert result['recall'] == 1.0 and result['false_positive_rate'] == .03


def test_svm_candidate_produces_bounded_spam_scores():
    from spamdet.training import spam_scores
    pipeline = make_pipeline('word_char_svm')
    texts = ['ordinary meeting today', 'ordinary lunch today']*4 + ['win money prize', 'claim money prize']*4
    pipeline.fit(texts, ['ham']*8 + ['spam']*8)
    assert hasattr(pipeline.named_steps['classifier'], 'calibrated_classifiers_')
    scores = spam_scores(pipeline, ['ordinary meeting today','win money prize'])
    assert all(0 <= s <= 1 for s in scores) and scores[1] > scores[0]
