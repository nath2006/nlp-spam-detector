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
