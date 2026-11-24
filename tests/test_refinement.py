import numpy as np
import pytest

from spamdet.refinement import select_group_threshold, allocate_groups


def test_group_constraint_cannot_hide_minority_ham_errors():
    labels=np.array(['ham']*1000+['spam']*60+['ham']*100+['spam']*40)
    scores=np.array([.1]*1000+[.7]*60+[.1]*95+[.8]*5+[.9]*40)
    groups=np.array(['sms_en']*1060+['sms_id']*140)
    threshold,report=select_group_threshold(labels,scores,groups)
    assert threshold>.8
    assert report['sms_id']['false_positive_rate']<=.03
    assert report['sms_en']['recall']==0  # trade-off must remain visible, not hidden


def test_small_groups_are_reported_without_compliance_claim():
    threshold,report=select_group_threshold(np.array(['ham']*100+['spam']*40),np.array([.1]*100+[.9]*40),np.array(['email_en']*140))
    assert threshold==.9 and report['email_en']['constraint_eligible']
    _,small=select_group_threshold(np.array(['ham']*3+['spam']*2),np.array([.1]*3+[.9]*2),np.array(['email_id']*5))
    assert not small['email_id']['constraint_eligible']


def test_new_holdout_never_reuses_old_groups_and_aug_is_training_only():
    rows=[dict(row_id='legacy',group='a',origin_split='train',source_id='history'),
          dict(row_id='overlap',group='a',source_id='india_sms'),
          dict(row_id='new',group='b',source_id='india_sms'),
          dict(row_id='aug',group='c',source_id='ots_legit_augmentation',synthetic=True)]
    result=allocate_groups(rows)
    by_id={row['row_id']:row for row in result}
    assert by_id['overlap']['split']=='excluded_overlap'
    assert by_id['new']['split'] in ('train','val','new_test')
    assert by_id['aug']['split']=='train'


def test_old_split_conflicts_are_quarantined_not_leaked():
    rows=[dict(row_id='a',group='merged',origin_split='train',source_id='history'),
          dict(row_id='b',group='merged',origin_split='test',source_id='history')]
    assert all(r['split']=='quarantine' for r in allocate_groups(rows))
