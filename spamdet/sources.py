"""Adapters for verified original archives. No network and no archive extraction."""
import csv
import io
import mailbox
from pathlib import Path
import tarfile
import zipfile

from spamdet.curation import parse_email


def read_source(path, source):
    rows = []

    def add(body, label, key, subject='', thread_ids=None, original_label=None):
        if label not in ('ham', 'spam'):
            raise ValueError('Unmapped label')
        # Same limits as inference; reject rather than silently truncate.
        if not body.strip() or len(body) > (10000 if source['channel'] == 'sms' else 100000) or len(subject) > 1000:
            return
        rows.append(dict(channel=source['channel'], language=source['language'], label=label,
                         body=body.strip(), subject=subject.strip(), source_id=source['id'],
                         row_id=source['id'] + ':' + str(key), thread_ids=thread_ids or [],
                         synthetic=False, train_only=source.get('train_only', False),
                         original_label=original_label or label))

    kind = source['format']
    if kind == 'uci_sms_zip':
        with zipfile.ZipFile(path) as archive:
            for i, line in enumerate(archive.read('SMSSpamCollection').decode('utf-8').splitlines(), 1):
                label, body = line.split('\t', 1)
                add(body, {'ham': 'ham', 'spam': 'spam'}[label], i)
    elif kind == 'mishra_zip':
        with zipfile.ZipFile(path) as archive:
            text = archive.read('Dataset_5971.csv').decode('utf-8-sig')
        for i, row in enumerate(csv.DictReader(io.StringIO(text)), 1):
            original = row['LABEL'].strip().lower()
            if original not in ('ham', 'spam', 'smishing'):
                raise ValueError('Unmapped Mishra label')
            add(row['TEXT'], 'ham' if original == 'ham' else 'spam', i, original_label=original)
    elif kind == 'yudi_sms_zip':
        with zipfile.ZipFile(path) as archive:
            text = archive.read('dataset_sms_spam_v1.csv').decode('utf-8-sig')
        for i, row in enumerate(csv.DictReader(io.StringIO(text)), 1):
            original = row['label'].strip()
            if original not in ('0', '1', '2'):
                raise ValueError('Unmapped Yudi label')
            add(row['Teks'], {'0': 'ham', '1': 'spam', '2': 'spam'}[original], i, original_label=original)
    elif kind in ('bopbi_tar', 'lingspam_tar'):
        with tarfile.open(path, 'r:gz') as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile() or member.size > 2 * 1024 * 1024:
                    continue
                name = Path(member.name)
                if kind == 'bopbi_tar':
                    if len(name.parts) != 3 or name.parts[1] not in ('loan', 'prize', 'premium-sms', 'evil-service', 'non-provider-promo') or name.suffix != '.txt':
                        continue
                    add(archive.extractfile(member).read().decode('utf-8'), 'spam', member.name)
                elif '/bare/' in member.name and name.suffix == '.txt':
                    raw = archive.extractfile(member).read()
                    subject, body, ids = parse_email(raw)
                    add(body, 'spam' if name.name.startswith('spmsg') else 'ham', member.name, subject, ids)
    elif kind == 'nazario_mbox':
        box = mailbox.mbox(str(path), create=False)
        try:
            for i, key in enumerate(box.iterkeys(), 1):
                subject, body, ids = parse_email(box.get_bytes(key))
                add(body, 'spam', i, subject, ids)
        finally:
            box.close()
    else:
        raise ValueError('Unknown source format: ' + kind)
    return rows
