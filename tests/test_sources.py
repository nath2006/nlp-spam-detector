import io
import json
import tarfile
import zipfile

import pytest

from spamdet.sources import read_source


def make_tar(path, files):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in files.items():
            data = content.encode('utf-8')
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_yudi_label_mapping_from_original_archive(tmp_path):
    path = tmp_path / 'sms.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('dataset_sms_spam_v1.csv', 'Teks,label\nhello,0\nfraud,1\npromo,2\n')
    rows = read_source(path, {'id': 'yudi_sms', 'format': 'yudi_sms_zip', 'channel': 'sms', 'language': 'id'})
    assert [r['label'] for r in rows] == ['ham', 'spam', 'spam']
    assert [r['original_label'] for r in rows] == ['0', '1', '2']
    assert all(r['language'] == 'id' and not r['synthetic'] for r in rows)


def test_yudi_rejects_unmapped_label(tmp_path):
    path = tmp_path / 'sms.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('dataset_sms_spam_v1.csv', 'Teks,label\nmessage,9\n')
    with pytest.raises(ValueError, match='label'):
        read_source(path, {'id': 'yudi_sms', 'format': 'yudi_sms_zip', 'channel': 'sms', 'language': 'id'})


def test_lingspam_reads_only_bare_and_preserves_optional_subject(tmp_path):
    path = tmp_path / 'mail.tar.gz'
    make_tar(path, {
        'lingspam_public/bare/part1/spmsg1.txt': 'Subject: offer\n\nclaim prize',
        'lingspam_public/bare/part1/3-1msg1.txt': 'Subject: meeting\n\nmeeting tomorrow',
        'lingspam_public/lemm/part1/spmsg1.txt': 'Subject: offer\n\nclaim prize',
    })
    rows = read_source(path, {'id': 'lingspam', 'format': 'lingspam_tar', 'channel': 'email', 'language': 'en'})
    assert len(rows) == 2
    assert {r['label'] for r in rows} == {'spam', 'ham'}
    spam = next(r for r in rows if r['label'] == 'spam')
    assert spam['subject'] == 'offer' and spam['body'] == 'claim prize'


def test_bopbi_single_class_is_explicit_training_only(tmp_path):
    path = tmp_path / 'id.tar.gz'
    make_tar(path, {'root/loan/1.txt': 'pinjaman cepat', 'root/LICENSE.txt': 'not a message'})
    rows = read_source(path, {'id': 'bopbi', 'format': 'bopbi_tar', 'channel': 'sms', 'language': 'id', 'train_only': True})
    assert len(rows) == 1
    assert rows[0]['label'] == 'spam' and rows[0]['train_only']


def test_mishra_uses_text_not_label_leaking_attributes(tmp_path):
    path = tmp_path / 'mishra.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('Dataset_5971.csv', 'LABEL,TEXT,URL,EMAIL,PHONE\nham,ordinary message,yes,yes,yes\nSmishing,claim account,No,No,No\nSpam,promo now,No,No,No\n')
    rows = read_source(path, {'id': 'mishra_sms', 'format': 'mishra_zip', 'channel': 'sms', 'language': 'en'})
    assert [r['label'] for r in rows] == ['ham','spam','spam']
    assert rows[0]['body'] == 'ordinary message'
