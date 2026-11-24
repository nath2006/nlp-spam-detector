import joblib
import numpy as np
import pytest
from spamdet.sms_recall import training_weights, promotion_gate, sms_only_bundle


def test_weights_balance_original_language_labels_and_discount_synthetic():
    rows=[dict(language='en',label='ham',synthetic=False)]*8+[
        dict(language='id',label='spam',synthetic=False)]*2+[
        dict(language='en',label='ham',synthetic=True)]
    weights=training_weights(rows,.1,True)
    assert weights[:8].sum()==pytest.approx(weights[8:10].sum())
    assert weights[10]==pytest.approx(weights[0]*.1)
    assert np.isfinite(weights).all() and (weights>0).all()


def group(fp,fn,ham=100,spam=100):
    return dict(false_positives=fp,false_negatives=fn,support={'ham':ham,'spam':spam},recall=(spam-fn)/spam,false_positive_rate=fp/ham)


def test_promotion_requires_recall_gain_without_domain_regression_and_low_fp():
    current={'id':group(0,10),'en':group(0,20)}
    assert promotion_gate(current,{'id':group(2,5),'en':group(1,10)})['accepted']
    assert not promotion_gate(current,{'id':group(4,0),'en':group(0,10)})['accepted']
    assert not promotion_gate(current,{'id':group(0,1),'en':group(0,21)})['accepted']
    assert not promotion_gate(current,current)['accepted']
    assert not promotion_gate(current,{'id':group(0,1)})['accepted']


def test_sms_replacement_preserves_email_and_does_not_mutate_original():
    original={'models':{'sms':{'name':'old'},'email':{'name':'email','threshold':.2}},'metadata':{'selection':'original'}}
    replacement={'name':'new'}
    updated=sms_only_bundle(original,replacement,{'accepted':True},{'round':'sms'})
    assert updated['models']['sms']==replacement
    assert joblib.hash(updated['models']['email'])==joblib.hash(original['models']['email'])
    assert original['models']['sms']['name']=='old'
    assert original['metadata']=={'selection':'original'}
    with pytest.raises(ValueError,match='gate'):
        sms_only_bundle(original,replacement,{'accepted':False},{})
