from spamdet.curation import canonical_text, parse_email, curate, assign_splits


def row(body, label='ham', channel='sms', **kw):
    return dict(channel=channel, language='en', label=label, subject='', body=body, source_id='test', row_id=body+label, thread_ids=[], **kw)


def test_canonical():
    assert canonical_text(row(' x ')) == 'x'
    assert canonical_text(row(' x ', channel='email')) == '\n\nx'


def test_email_no_header_leakage():
    subject, body, ids = parse_email(b'Subject: Hello\nX-Spam-Status: YES\nMessage-ID: <a>\nContent-Type: text/html\n\n<p>World</p><script>evil()</script>')
    assert subject == 'Hello' and 'World' in body
    assert 'YES' not in body and 'evil' not in body and ids == ['<a>']


def test_duplicates_conflicts_and_templates():
    rows = [row('hello'), row('hello'), row('conflict'), row('conflict','spam'), row('win prize call 12345','spam'), row('win prize call 99999','spam')]
    result, audit = curate(rows)
    assert len(result) == 3
    assert audit['exact_duplicates_removed'] == 1
    assert audit['conflicting_rows_removed'] == 2
    assert result[1]['group'] == result[2]['group']


def test_group_split_is_reproducible_disjoint():
    rows = [row('distinct text '+chr(0x400+i), 'spam' if i%3 == 0 else 'ham') for i in range(90)]
    for i,r in enumerate(rows):
        r['group'] = str(i//2)
    a = assign_splits(rows)
    b = assign_splits(rows)
    assert a == b
    groups = {s:{r['group'] for r in a if r['split']==s} for s in ['train','val','test']}
    assert not groups['train'] & groups['test']
    assert not groups['val'] & groups['test']
    assert not groups['train'] & groups['val']
