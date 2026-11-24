"""Download pinned public archives with verified TLS via curl; never run archive code."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def download_source(source, root=ROOT):
    target = (root / source['path']).resolve()
    allowed = (root / 'data/raw').resolve()
    if not target.is_relative_to(allowed):
        raise ValueError('Archive must be inside data/raw')
    expected = source['sha256']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError('Checksum mismatch for existing archive: ' + str(target))
        print(source['id'] + ': verified cache', flush=True)
        return 'cached'
    executable = shutil.which('curl.exe') or shutil.which('curl')
    if not executable:
        raise RuntimeError('curl is required. Install curl or download the manifest URL manually.')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    subprocess.run([executable, '--fail', '--location', '--connect-timeout', '20', '--max-time', '240',
                    '--proto', '=https', '--proto-redir', '=https', source['download_url'], '--output', str(partial)],
                   check=True, timeout=250)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
        raise ValueError('Checksum mismatch in downloaded archive: ' + str(partial))
    partial.replace(target)
    print(source['id'] + ': downloaded and verified', flush=True)
    return 'downloaded'


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='data/sources.json')
    args=parser.parse_args(argv)
    manifest = json.loads((ROOT / args.manifest).read_text(encoding='utf-8'))
    for source in manifest['sources']:
        download_source(source)


if __name__ == '__main__':
    main()
