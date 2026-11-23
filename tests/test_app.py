import builtins
from pathlib import Path
import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')

@pytest.fixture
def app(tmp_path, monkeypatch):
    pipe = Pipeline([('features', TfidfVectorizer()), ('classifier', LogisticRegression())])
    pipe.fit(['hello meeting', 'friend lunch', 'win prize', 'free cash'], ['ham','ham','spam','spam'])
    path = tmp_path/'synthetic.joblib'
    joblib.dump({'models': {c: {'pipeline':pipe,'threshold':.5,'name':'synthetic'} for c in ('sms','email')}, 'metadata':{}}, path)
    monkeypatch.setenv('SPAM_MODEL_PATH', str(path))
    return AppTest.from_file(APP).run()


def test_navigation_no_ocr(app, monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name in ('cv2','easyocr'):
            raise ImportError('OCR unavailable')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    app.run()
    assert not app.exception
    assert app.radio[0].options == ['Deteksi SMS','Deteksi Email','Deteksi Massal','Evaluasi Model']
    assert any('belum tervalidasi' in w.value for w in app.warning)
    app.text_area(key='sms_body').set_value('hello friend')
    app.button(key='sms_detect').click().run()
    assert not app.exception
    assert any(m.label == 'Skor model' for m in app.metric)


def test_blank_and_long(app):
    app.button(key='sms_detect').click().run()
    assert app.error
    app.text_area(key='sms_body').set_value('x'*10001)
    app.button(key='sms_detect').click().run()
    assert app.error and not app.exception


def test_email_no_subject(app):
    app.radio[0].set_value('Deteksi Email').run()
    app.text_area(key='email_body').set_value('meeting tomorrow 你好')
    app.button(key='email_detect').click().run()
    assert app.metric and not app.exception


def test_examples(app):
    assert len(app.selectbox[0].options) == 4
    app.button(key='sms_fill').click().run()
    assert app.text_area(key='sms_body').value


def test_batch_template(app):
    app.radio[0].set_value('Deteksi Massal').run()
    app.button(key='batch_example').click().run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 8
    assert 'body' not in app.dataframe[0].value.columns


def test_eval(app):
    app.radio[0].set_value('Evaluasi Model').run()
    assert not app.exception


def test_eval_missing_and_actual_schema(app, monkeypatch):
    original = Path.read_text
    def missing(path, *args, **kwargs):
        if path.name == 'evaluation.json':
            raise FileNotFoundError()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', missing)
    app.radio[0].set_value('Evaluasi Model').run()
    assert app.info and not app.exception
    def actual(path, *args, **kwargs):
        if path.name == 'evaluation.json':
            return '{"sms": {"test": {"f1": 0.321}}, "metadata": {"synthetic": true}}'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', actual)
    app.run()
    assert not app.exception
    assert app.json and any('0.321' in item.value for item in app.json)


def test_partial_model_ui(app, monkeypatch):
    import os
    path = os.environ['SPAM_MODEL_PATH']
    bundle = joblib.load(path)
    del bundle['models']['email']
    joblib.dump(bundle, path)
    app.radio[0].set_value('Deteksi Email').run()
    assert any('email belum tersedia' in item.value for item in app.info)
    app.text_area(key='email_body').set_value('meeting tomorrow')
    app.button(key='email_detect').click().run()
    assert any('email belum tersedia' in item.value for item in app.error)
    app.radio[0].set_value('Deteksi SMS').run()
    app.text_area(key='sms_body').set_value('hello meeting')
    app.button(key='sms_detect').click().run()
    assert app.metric and not app.exception
    app.radio[0].set_value('Deteksi Massal').run()
    app.button(key='batch_example').click().run()
    rows = app.dataframe[0].value
    assert rows.error.isna().sum() == 4
    assert rows.error.notna().sum() == 4


def test_eval_table_and_unvalidated_groups(app):
    app.radio[0].set_value('Evaluasi Model').run()
    assert app.dataframe
    table = app.dataframe[0].value
    assert set(table['Kelompok']) == {'SMS Inggris','SMS Indonesia','Email Inggris','Email Indonesia'}
    assert any('belum tervalidasi' in str(x).lower() for x in table['Status'])
    assert any('UCI' in x.value for x in app.text)


def test_missing_model(app, monkeypatch, tmp_path):
    monkeypatch.setenv('SPAM_MODEL_PATH', str(tmp_path/'missing'))
    app.text_area(key='sms_body').set_value('hello')
    app.button(key='sms_detect').click().run()
    assert not app.exception
    assert any('tidak ditemukan' in e.value for e in app.error)
