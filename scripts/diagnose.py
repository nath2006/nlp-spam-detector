"""Manual-only diagnostics: never imported by data preparation or model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from spamdet.inference import DEFAULT_ARTIFACT, ModelError, canonical_input, predict
from spamdet.curation import canonical_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--artifact')
    args = parser.parse_args()
    path = Path(args.artifact or os.environ.get('SPAM_MODEL_PATH') or DEFAULT_ARTIFACT).resolve()
    bundle = joblib.load(path)
    report = {'artifact': str(path), 'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'metadata': bundle['metadata'], 'models': {}, 'manual_only': True, 'results': []}
    for channel, entry in bundle['models'].items():
        report['models'][channel] = {'name': entry['name'], 'threshold': entry['threshold'],
                                    'classes': list(entry['pipeline'].classes_),
                                    'spam_index': list(entry['pipeline'].classes_).index('spam')}
    os.environ['SPAM_MODEL_PATH'] = str(path)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / 'app.py')).run()
    assert not app.exception
    cases = json.loads((ROOT / 'examples/diagnostic_messages.json').read_text(encoding='utf-8'))
    for case in cases:
        channel, body, subject = case['channel'], case['body'], case.get('subject', '')
        assert canonical_input(channel, body, subject) == canonical_text(dict(case, subject=subject))
        row = {'id': case['id'], 'language': case['language'], 'channel': channel, 'expected': case['expected']}
        try:
            result = predict(channel, body, subject, artifact_path=path)
            row.update(result)
        except ModelError as exc:
            row['error'] = str(exc)
        app.radio[0].set_value('Deteksi SMS' if channel == 'sms' else 'Deteksi Email').run()
        app.text_area(key=channel + '_body').set_value(body)
        if channel == 'email':
            app.text_input(key='email_subject').set_value(subject)
        app.button(key=channel + '_detect').click().run()
        assert not app.exception
        if row.get('error'):
            assert any(row['error'] == item.value for item in app.error)
        else:
            assert any(item.value == row['display'] for item in app.subheader)
            assert app.metric[0].value == f"{row['score_model']:.4f}"
            assert app.metric[1].value == f"{row['threshold']:.4f}"
        row['ui_direct_parity'] = True
        report['results'].append(row)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
