"""Offline source parsing, leakage-safe canonicalization and group partitioning."""
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026


def canonical_text(row):
    body = row['body'].strip()
    return row['subject'].strip() + '\n\n' + body if row['channel'] == 'email' else body


def parse_email(raw):
    """Only decoded Subject + MIME textual body; headers used ONLY for grouping."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    pieces = []
    # Prefer text/plain alternative; no attachments or embedded forwarded messages.
    def extract(part):
        if part.get_content_disposition() == 'attachment' or part.get_content_type() == 'message/rfc822':
            return
        if part.is_multipart():
            children = list(part.iter_parts())
            if part.get_content_subtype() == 'alternative':
                children = [next((p for p in children if p.get_content_type() == 'text/plain'), children[-1])]
            for child in children:
                extract(child)
        elif part.get_content_type() in ('text/plain', 'text/html'):
            try:
                content = part.get_content()
            except (LookupError, UnicodeError):
                content = (part.get_payload(decode=True) or b'').decode('utf-8', errors='replace')
            if part.get_content_type() == 'text/html':
                soup = BeautifulSoup(content, 'html.parser')
                for el in soup(['script', 'style', 'noscript', 'template']):
                    el.decompose()
                content = soup.get_text(' ', strip=True)
            pieces.append(str(content).strip())
    extract(msg)
    ids = re.findall(r'<[^>]+>', ' '.join(str(msg.get(k, '')) for k in ('Message-ID', 'References', 'In-Reply-To')))
    return str(msg.get('Subject', '')).strip(), '\n'.join(pieces).strip(), ids


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def template(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', normalized(text))
    text = re.sub(r'[\w.+-]+@[\w.-]+', ' EMAIL ', text)
    return re.sub(r'\W+', ' ', re.sub(r'\d+', ' NUM ', text)).strip()


def curate(rows):
    """Global exact conflicts/dedup before channel splitting; group cosine >= .90."""
    by_text = defaultdict(list)
    empty = 0
    for row in rows:
        if row['label'] not in ('ham', 'spam'):
            raise ValueError('Unmapped label')
        if not row['body'].strip() and not row['subject'].strip():
            empty += 1
            continue
        by_text[normalized(canonical_text(row))].append(row)
    retained, duplicates, conflicts = [], 0, 0
    for key, batch in by_text.items():
        if len({r['label'] for r in batch}) > 1:
            conflicts += len(batch)
            continue
        duplicates += len(batch)-1
        r = dict(batch[0])
        r['duplicate_source_ids'] = sorted({x['source_id'] for x in batch})
        r['thread_ids'] = sorted({x for b in batch for x in b.get('thread_ids', [])})
        r['text_hash'] = digest(key)
        retained.append(r)
    parent = list(range(len(retained)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        a,b = root(i), root(j)
        parent[max(a,b)] = min(a,b)
    templates, threads = {}, {}
    for i,r in enumerate(retained):
        key = template(canonical_text(r))
        if key in templates:
            union(i,templates[key])
        templates[key] = i
        for t in r.get('thread_ids', []):
            if t in threads:
                union(i, threads[t])
            threads[t] = i
    # Character TF-IDF only for UNSUPERVISED grouping, never reused by classifier.
    # Exclude very short texts from fuzzy grouping: exact/template still applies.
    eligible = [i for i,r in enumerate(retained) if len(canonical_text(r)) >= 40]
    near_edges = 0
    if len(eligible) > 1:
        vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=20000).fit_transform([canonical_text(retained[i]) for i in eligible])
        nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=1).fit(vectors)
        for start in range(0,len(eligible),128):
            distances, neighbors = nn.radius_neighbors(vectors[start:start+128], radius=.10000001)
            for offset,(dd,jj) in enumerate(zip(distances,neighbors)):
                i = start+offset
                for distance,j in zip(dd,jj):
                    if j > i:
                        union(eligible[i],eligible[j])
                        near_edges += 1
    for i,r in enumerate(retained):
        r['group'] = retained[root(i)]['text_hash']
    return retained, dict(raw_rows=len(rows), retained_rows=len(retained), empty_removed=empty,
                         exact_duplicates_removed=duplicates, conflicting_rows_removed=conflicts,
                         near_duplicate_edges=near_edges, groups=len({r['group'] for r in retained}),
                         largest_group=max(Counter(r['group'] for r in retained).values(),default=0))


def assign_splits(rows):
    # Five grouped stratified folds yield approximately 60/20/20, preserving groups
    # globally across source/channel/language boundaries.
    y = [r['channel']+'_'+r['language']+'_'+r['label'] for r in rows]
    groups = [r['group'] for r in rows]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignments = {}
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(rows)),y,groups)):
        for i in held:
            assignments[int(i)] = 'test' if fold == 0 else 'val' if fold == 1 else 'train'
    return [dict(r, split=assignments[i]) for i,r in enumerate(rows)]


def prepare():
    source = ROOT/'data/raw/uci_sms.zip'
    with zipfile.ZipFile(source) as archive:
        lines = archive.read('SMSSpamCollection').decode('utf-8').splitlines()
    rows = []
    for i,line in enumerate(lines):
        label, body = line.split('\t',1)
        rows.append(dict(channel='sms', language='en', label={'ham':'ham','spam':'spam'}[label],
                         subject='', body=body, source_id='uci_sms', row_id=f'uci_sms:{i+1}',thread_ids=[]))
    # Optional official email corpus is intentionally not fabricated when access is denied.
    result, audit = curate(rows)
    result = assign_splits(result)
    report = dict(schema_version=1, seed=SEED, sources={'uci_sms':{
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_counts':dict(Counter(r['label'] for r in rows)),
        'curated_counts':dict(Counter(r['label'] for r in result)),
        'license':'CC BY 4.0','attribution':'Almeida & Hidalgo; UCI SMS Spam Collection; DOI 10.24432/C5CC84'}},
        audit=audit, splits={s:dict(Counter(r['channel']+'_'+r['language']+'_'+r['label'] for r in result if r['split']==s)) for s in ['train','val','test']},
        split_sha256=digest('\n'.join(r['row_id']+':'+r['group']+':'+r['split'] for r in result)),
        grouping='Global normalized exact dedup; all conflicting exact labels excluded; URL/email/number-normalized templates; char 3-5 cosine >=0.90; transitive union plus thread IDs; five stratified group folds 60/20/20.',
        sample_audit=[{'row_id':next(r['row_id'] for r in result if r['label']==label), 'label':label,
                       'note':note} for label,note in [('ham','Informal conversational English SMS; no email headers.'),('spam','Prize/premium-call promotion SMS; original spam label retained.')]],
        language_policy='Source-level en designation; informal English/mixed varieties, not individually verified monolingual English. Indonesian unvalidated.',
        email={'status':'unavailable','reason':'Official SpamAssassin HTTP retrieval command denied by user/tool approval layer. Not retried, not bypassed. No local email data; no email training.'},
        independent_source_test='Not performed: only one eligible accessible source, with unavailable component source IDs.',
        privacy='Raw and processed data local/ignored; no full messages or phone numbers in reports.')
    for path in [ROOT/'data/processed',ROOT/'reports']:
        path.mkdir(parents=True,exist_ok=True)
    (ROOT/'data/processed/curated.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    (ROOT/'reports/curation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report
