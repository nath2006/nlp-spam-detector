"""Synthetic fixtures exercise inference, never claim real-world performance."""
import csv
import io
import json

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from spamdet.inference import predict, predict_csv, export_csv, canonical_input, ModelError


@pytest.fixture
def artifact(tmp_path):
    pipe = Pipeline([('features', TfidfVectorizer()), ('classifier', LogisticRegression())])
    pipe.fit(['meeting tomorrow office', 'hello friend lunch', 'win prize cash now', 'free cash winner'], ['ham', 'ham', 'spam', 'spam'])
    path = tmp_path / 'model.joblib'
    joblib.dump({'models': {c: {'pipeline': pipe, 'threshold': .6, 'name': 'synthetic'} for c in ('sms', 'email')}, 'metadata': {'synthetic': True}}, path)
    return path


@pytest.mark.parametrize('channel,body,subject', [('sms', '', ''), ('email', '  ', 'subject'), ('SMS', 'hello', ''), ('sms', None, ''), ('sms', 'a'*10001, ''), ('email', 'a'*100001, ''), ('email', 'hello', 'a'*1001), ('sms', 'hi', None)])
def test_invalid_input(artifact, channel, body, subject):
    with pytest.raises(ValueError):
        predict(channel, body, subject, artifact_path=artifact)


def test_canonical():
    assert canonical_input('sms', ' hi ', '') == 'hi'
    assert canonical_input('email', ' hi ', '') == '\n\nhi'
    assert canonical_input('email', ' hi ', ' subject ') == 'subject\n\nhi'


@pytest.mark.parametrize('body', ['hello friend', 'Selamat siang teman', '你好 👋 مرحبا', '<script>alert(1)</script>'])
def test_serializable_no_raw(artifact, body):
    result = predict('sms', body, artifact_path=artifact)
    assert result['label'] in ('ham', 'spam')
    assert result['label'] == ('spam' if result['score_model'] >= .6 else 'ham')
    assert body not in json.dumps(result)
    assert result['error'] is None


def test_email_no_subject(artifact):
    assert predict('email', 'meeting', artifact_path=artifact)['error'] is None


def test_missing_corrupt(tmp_path):
    with pytest.raises(ModelError, match='tidak ditemukan'):
        predict('sms', 'hello', artifact_path=tmp_path/'absent')
    path = tmp_path/'bad'
    path.write_bytes(b'broken')
    with pytest.raises(ModelError, match='rusak|valid'):
        predict('sms', 'hello', artifact_path=path)


def test_env_and_default(artifact, monkeypatch, tmp_path):
    monkeypatch.setenv('SPAM_MODEL_PATH', str(artifact))
    monkeypatch.chdir(tmp_path)
    assert predict('sms', 'hello')['label'] in ('ham', 'spam')
    monkeypatch.delenv('SPAM_MODEL_PATH')
    from spamdet.inference import DEFAULT_ARTIFACT
    assert DEFAULT_ARTIFACT.is_absolute()
    assert DEFAULT_ARTIFACT.name == 'model.joblib'


@pytest.mark.parametrize('nb,union', [(False, False), (True, False), (False, True)])
def test_real_contributions(artifact, nb, union):
    bundle = joblib.load(artifact)
    features = FeatureUnion([('words', TfidfVectorizer()), ('chars', TfidfVectorizer(analyzer='char', ngram_range=(2, 3)))]) if union else TfidfVectorizer()
    pipe = Pipeline([('features', features), ('classifier', MultinomialNB() if nb else LogisticRegression())])
    pipe.fit(['hello meeting office', 'hello friend', 'free prize cash', 'winner cash now'], ['ham', 'ham', 'spam', 'spam'])
    bundle['models']['sms']['pipeline'] = pipe
    joblib.dump(bundle, artifact)
    result = predict('sms', 'free cash hello', artifact_path=artifact, explain=True)
    exp = result['explanation']
    assert exp['intercept'] + exp['sum_contributions'] == pytest.approx(exp['decision_margin'])
    assert sum(x['contribution'] for x in exp['features']) == pytest.approx(exp['sum_contributions'])
    assert 1/(1+np.exp(-exp['decision_margin'])) == pytest.approx(result['score_model'])
    assert exp['threshold_margin'] == pytest.approx(np.log(.6/.4))


def test_csv_mixed(artifact):
    rows = predict_csv(b'channel,body,subject\nsms,hello,\nemail,meeting,\nsms,,x\ninvalid,hello,\n', artifact)
    assert len(rows) == 4
    assert rows[0]['row'] == 2
    assert not rows[0]['error'] and rows[2]['error'] and rows[3]['error']
    assert all('body' not in row for row in rows)


@pytest.mark.parametrize('data', [b'', b'body\nhi\n', b'channel,body,body\nsms,hi,hi\n', b'channel,body\nsms,\xff', b'x'* (2*1024*1024+1), b'channel,body\n'+b'sms,hi\n'*1001, b'channel,body\nsms,"unterminated'])
def test_bad_csv(artifact, data):
    with pytest.raises(ValueError):
        predict_csv(data, artifact)


def test_extra_fields(artifact):
    result = predict_csv(b'channel,body\nsms,hello,extra\nsms,hello\n', artifact)
    assert result[0]['error'] and not result[1]['error']


def test_csv_missing_cells_and_empty(artifact):
    rows = predict_csv(b'channel,body,subject\nsms\nemail,hello,\n', artifact)
    assert rows[0]['error'] and rows[1]['error'] is None
    with pytest.raises(ValueError):
        predict_csv(b'channel,body\n', artifact)


def test_corrupt_schema(artifact):
    joblib.dump({'models': {}, 'metadata': {}}, artifact)
    with pytest.raises(ModelError, match='valid'):
        predict('sms', 'hello', artifact_path=artifact)


def test_sms_only_artifact_and_missing_email(artifact):
    bundle = joblib.load(artifact)
    del bundle['models']['email']
    joblib.dump(bundle, artifact)
    assert predict('sms', 'hello meeting', artifact_path=artifact)['error'] is None
    with pytest.raises(ModelError, match='email.*belum tersedia'):
        predict('email', 'hello meeting', artifact_path=artifact)
    rows = predict_csv('channel,body\nsms,hello\nemail,hello\nsms,\n', artifact)
    assert rows[0]['error'] is None
    assert 'email' in rows[1]['error'] and rows[2]['error']


def test_channel_status(artifact):
    from spamdet.inference import model_status
    assert model_status('sms', artifact)['available'] is True
    bundle = joblib.load(artifact)
    del bundle['models']['email']
    joblib.dump(bundle, artifact)
    assert model_status('sms', artifact)['available'] is True
    assert model_status('email', artifact)['available'] is False
    assert 'belum tersedia' in model_status('email', artifact)['message']


def test_status_identifies_actual_artifact_and_reloads_replacement(artifact):
    import hashlib
    from spamdet.inference import model_status
    status = model_status('sms', artifact)
    assert status['artifact_path'] == str(artifact.resolve())
    assert status['artifact_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert status['classes'] == ['ham','spam'] and status['spam_index'] == 1
    assert status['threshold'] == .6
    bundle = joblib.load(artifact)
    bundle['models']['sms']['threshold'] = .75
    joblib.dump(bundle, artifact)
    updated = model_status('sms', artifact)
    assert updated['threshold'] == .75
    assert updated['artifact_sha256'] != status['artifact_sha256']


def test_export_all_strings():
    row = {key: value for key, value in zip(('row','label','display','score_model','threshold','error'), ['=1','+cmd','-cmd','@SUM(1)','\t=1','  =1'])}
    exported = list(csv.DictReader(io.StringIO(export_csv([row]))))[0]
    assert all(value.startswith("'") for value in exported.values())
