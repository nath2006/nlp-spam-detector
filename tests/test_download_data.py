import hashlib
import pytest
from scripts.download_data import download_source


def test_verified_existing_archive_skips_network(tmp_path):
    path = tmp_path/'data/raw/archive.zip'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'archive')
    spec = dict(path='data/raw/archive.zip', sha256=hashlib.sha256(b'archive').hexdigest(), download_url='https://example.invalid/archive.zip', id='test')
    assert download_source(spec, tmp_path) == 'cached'


def test_corrupt_existing_archive_does_not_get_silently_overwritten(tmp_path):
    path = tmp_path/'data/raw/archive.zip'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupt')
    spec = dict(path='data/raw/archive.zip', sha256=hashlib.sha256(b'archive').hexdigest(), download_url='https://example.invalid/archive.zip', id='test')
    with pytest.raises(ValueError, match='Checksum'):
        download_source(spec, tmp_path)
    assert path.read_bytes() == b'corrupt'


def test_cli_accepts_explicit_additional_source_manifest(tmp_path,monkeypatch):
    import json
    import scripts.download_data as module
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'sources':[{'id':'additional'}]}))
    calls=[]
    monkeypatch.setattr(module,'download_source',lambda source:calls.append(source['id']))
    module.main(['--manifest',str(path)])
    assert calls==['additional']
