"""Real-artifact integration checks; manual fixtures never train/select a model.

Skipped on a fresh checkout until download/prepare/train have produced the local
artifact. These checks test application behavior, not a representative quality benchmark.
"""
import hashlib
import json
from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest
from spamdet.inference import predict, predict_csv
from spamdet.curation import canonical_text, normalized

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts/model.joblib'
pytestmark = pytest.mark.skipif(not ARTIFACT.exists(), reason='Run offline training for real-artifact tests')


def test_real_artifact_matches_reports_and_has_email():
    bundle = joblib.load(ARTIFACT)
    report = json.loads((ROOT/'reports/evaluation.json').read_text(encoding='utf-8'))
    assert set(bundle['models']) == {'sms','email'}
    assert not bundle['metadata'].get('synthetic', False)
    assert bundle['metadata']['sources']
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == report['artifact_sha256']
    assert bundle['metadata']['languages_by_channel'] == {'email':['en'],'sms':['en','id']}


@pytest.mark.parametrize('case_index', range(4))
def test_manual_sms_regression_and_real_ui_parity(case_index, monkeypatch):
    monkeypatch.setenv('SPAM_MODEL_PATH', str(ARTIFACT))
    case = json.loads((ROOT/'examples/diagnostic_messages.json').read_text(encoding='utf-8'))[case_index]
    direct = predict('sms', case['body'], artifact_path=ARTIFACT)
    assert direct['label'] == 'spam'
    app = AppTest.from_file(str(ROOT/'app.py')).run()
    app.text_area(key='sms_body').set_value(case['body'])
    app.button(key='sms_detect').click().run()
    assert not app.exception
    assert app.metric[0].value == f"{direct['score_model']:.4f}"
    assert any(item.value == direct['display'] for item in app.subheader)


@pytest.mark.parametrize('subject', ['', 'Account verification'])
def test_real_email_ui_with_optional_subject(subject, monkeypatch):
    monkeypatch.setenv('SPAM_MODEL_PATH', str(ARTIFACT))
    body = 'Your account will be suspended. Send your password and verification code immediately.'
    direct = predict('email', body, subject, artifact_path=ARTIFACT)
    app = AppTest.from_file(str(ROOT/'app.py')).run()
    app.radio[0].set_value('Deteksi Email').run()
    app.text_input(key='email_subject').set_value(subject)
    app.text_area(key='email_body').set_value(body)
    app.button(key='email_detect').click().run()
    assert not app.exception and not app.error
    assert app.metric[0].value == f"{direct['score_model']:.4f}"
    assert any('Model email tersedia' in item.value for item in app.info)


def test_real_mixed_csv_ui_has_predictions_for_both_channels(monkeypatch):
    monkeypatch.setenv('SPAM_MODEL_PATH', str(ARTIFACT))
    direct = predict_csv((ROOT/'examples/batch_template.csv').read_bytes(), ARTIFACT)
    assert len(direct) == 8 and all(row['error'] is None for row in direct)
    app = AppTest.from_file(str(ROOT/'app.py')).run()
    app.radio[0].set_value('Deteksi Massal').run()
    app.button(key='batch_example').click().run()
    assert not app.exception
    table = app.dataframe[0].value
    assert table['label'].tolist() == [row['label'] for row in direct]
    assert table['score_model'].tolist() == pytest.approx([row['score_model'] for row in direct])


def test_real_group_holdout_and_manual_exclusion():
    rows = [json.loads(line) for line in (ROOT/'data/processed/curated.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], set()).add(row['split'])
        if row.get('train_only') or row.get('synthetic'):
            assert row['split'] == 'train'
    assert all(len(values) == 1 for values in groups.values())
    manual = json.loads((ROOT/'examples/diagnostic_messages.json').read_text(encoding='utf-8'))
    originals = {normalized(canonical_text(r)) for r in rows}
    assert all(normalized(canonical_text(dict(case, subject=case.get('subject','')))) not in originals for case in manual)
