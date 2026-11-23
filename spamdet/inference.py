"""Private, in-memory inference. Load only trusted, locally trained joblib artifacts."""
import csv
import io
import math
import os
from pathlib import Path

import joblib
import numpy as np

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / 'artifacts' / 'model.joblib'
FIELDS = ('row', 'label', 'display', 'score_model', 'threshold', 'error')


class ModelError(RuntimeError):
    """Model unavailable or incompatible; no input is included in diagnostics."""


def canonical_input(channel, body, subject=''):
    if not isinstance(channel, str) or channel not in ('sms', 'email'):
        raise ValueError('Channel harus sms atau email.')
    if not isinstance(body, str) or not body.strip():
        raise ValueError('Isi pesan wajib berupa teks dan tidak boleh kosong.')
    if not isinstance(subject, str):
        raise ValueError('Subjek harus berupa teks.')
    limit = 10000 if channel == 'sms' else 100000
    if len(body) > limit:
        raise ValueError(f'Isi pesan maksimal {limit} karakter.')
    if len(subject) > 1000:
        raise ValueError('Subjek maksimal 1000 karakter.')
    return body.strip() if channel == 'sms' else subject.strip() + '\n\n' + body.strip()


def _load(artifact_path):
    path = Path(artifact_path or os.environ.get('SPAM_MODEL_PATH') or DEFAULT_ARTIFACT)
    if not path.is_file():
        raise ModelError('Model tidak ditemukan. Jalankan pelatihan offline atau atur SPAM_MODEL_PATH.')
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle['metadata'], dict):
            raise ValueError()
        models = bundle['models']
        if not isinstance(models, dict) or not models or not set(models) <= {'sms', 'email'}:
            raise ValueError()
        for entry in models.values():
            pipe = entry['pipeline']
            if set(pipe.classes_) != {'ham', 'spam'} or not isinstance(entry['name'], str):
                raise ValueError()
            if not math.isfinite(float(entry['threshold'])) or not 0 <= float(entry['threshold']) <= 1:
                raise ValueError()
            if 'features' not in pipe.named_steps or 'classifier' not in pipe.named_steps:
                raise ValueError()
        return bundle
    except Exception:
        raise ModelError('Artefak model rusak atau format tidak valid. Latih ulang model offline.') from None


def _explain(pipe, text, threshold):
    features = pipe.named_steps['features']
    classifier = pipe.named_steps['classifier']
    x = features.transform([text]).tocsr()
    names = features.get_feature_names_out()
    classes = list(classifier.classes_)
    spam, ham = classes.index('spam'), classes.index('ham')
    if hasattr(classifier, 'coef_'):
        sign = 1 if spam == 1 else -1
        weights = classifier.coef_[0] * sign
        intercept = float(classifier.intercept_[0] * sign)
    elif hasattr(classifier, 'feature_log_prob_'):
        weights = classifier.feature_log_prob_[spam] - classifier.feature_log_prob_[ham]
        intercept = float(classifier.class_log_prior_[spam] - classifier.class_log_prior_[ham])
    else:
        return {'available': False, 'reason': 'Penjelasan linear tidak tersedia untuk model ini.'}
    contributions = [{'feature': str(names[i]), 'contribution': float(v * weights[i])} for i, v in zip(x.indices, x.data)]
    total = float(sum(item['contribution'] for item in contributions))
    contributions.sort(key=lambda item: abs(item['contribution']), reverse=True)
    # None represents an infinite logit at endpoint thresholds, retaining strict JSON support.
    boundary = math.log(threshold / (1-threshold)) if 0 < threshold < 1 else None
    return {'available': True, 'features': contributions, 'intercept': intercept,
            'sum_contributions': total, 'decision_margin': intercept + total,
            'threshold_margin': boundary, 'threshold': threshold,
            'threshold_endpoint': ('always_spam' if threshold == 0 else 'score_must_equal_one') if boundary is None else None}


def model_status(channel, artifact_path=None):
    """Inspect availability without classifying or retaining any message."""
    if channel not in ('sms', 'email'):
        raise ValueError('Channel harus sms atau email.')
    try:
        bundle = _load(artifact_path)
    except ModelError as exc:
        return {'available': False, 'message': str(exc)}
    if channel not in bundle['models']:
        return {'available': False, 'message': f'Model {channel} belum tersedia. Persiapkan korpus berizin dan latih model kanal ini secara offline.'}
    return {'available': True, 'message': f"Model {channel} tersedia: {bundle['models'][channel]['name']}."}


def _predict(bundle, channel, text, row=1, explain=False):
    if channel not in bundle['models']:
        raise ModelError(f'Model {channel} belum tersedia. Tidak menggunakan model SMS untuk email.')
    entry = bundle['models'][channel]
    pipe = entry['pipeline']
    try:
        score = float(pipe.predict_proba([text])[0][list(pipe.classes_).index('spam')])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError()
        threshold = float(entry['threshold'])
        label = 'spam' if score >= threshold else 'ham'
        result = dict(row=row, label=label, display='SPAM' if label == 'spam' else 'BUKAN SPAM', score_model=score, threshold=threshold, error=None)
        if explain:
            result['explanation'] = _explain(pipe, text, threshold)
        return result
    except Exception:
        raise ModelError('Artefak model tidak valid atau tidak kompatibel untuk inferensi.') from None


def predict(channel, body, subject='', artifact_path=None, *, explain=False):
    text = canonical_input(channel, body, subject)
    return _predict(_load(artifact_path), channel, text, explain=explain)


def _error(row, message):
    return dict(row=row, label=None, display=None, score_model=None, threshold=None, error=message)


def predict_csv(content, artifact_path=None):
    if isinstance(content, str):
        try:
            content = content.encode('utf-8')
        except UnicodeError:
            raise ValueError('CSV harus berupa UTF-8 yang valid.') from None
    if not isinstance(content, bytes):
        raise ValueError('CSV harus berupa bytes atau teks UTF-8.')
    if len(content) > 2 * 1024 * 1024:
        raise ValueError('CSV maksimal 2 MiB.')
    try:
        text = content.decode('utf-8-sig')
        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or not {'channel', 'body'} <= set(header):
            raise ValueError('Header CSV wajib channel,body; subject opsional, tanpa header duplikat.')
        records = []
        for fields in reader:
            if len(records) >= 1000:
                raise ValueError('CSV maksimal 1000 baris data.')
            records.append(fields)
    except (UnicodeError, csv.Error, StopIteration):
        raise ValueError('CSV kosong, UTF-8 tidak valid, atau format CSV rusak.') from None
    if not records:
        raise ValueError('CSV harus memuat minimal satu baris data.')
    bundle = _load(artifact_path)
    results = []
    for row, fields in enumerate(records, 2):
        if len(fields) != len(header):
            results.append(_error(row, 'Jumlah kolom baris tidak sesuai header.'))
            continue
        record = dict(zip(header, fields))
        try:
            canonical = canonical_input(record['channel'], record['body'], record.get('subject', ''))
            results.append(_predict(bundle, record['channel'], canonical, row=row))
        except (ValueError, ModelError) as exc:
            results.append(_error(row, str(exc)))
    return results


def safe_cell(value):
    """Neutralize spreadsheet formula prefixes on every untrusted string field."""
    if isinstance(value, str):
        stripped = value.lstrip()
        if (stripped and stripped[0] in '=+-@') or value.startswith(('\t', '\r', '\n')):
            return "'" + value
    return value


def export_csv(results):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()
    for row in results:
        writer.writerow({key: safe_cell(row.get(key)) for key in FIELDS})
    return output.getvalue()
