"""SMS-only recall experiments; helpers do not access manual diagnosis fixtures."""
from collections import Counter
import numpy as np


def training_weights(rows, synthetic_weight=1., balance_language_label=False):
    if not 0 < synthetic_weight <= 1:
        raise ValueError('Synthetic weight must be in (0,1].')
    counts = Counter((r['language'],r['label']) for r in rows if not r.get('synthetic'))
    weights = np.array([1./counts.get((r['language'],r['label']),1)
                        if balance_language_label else 1. for r in rows],dtype=float)
    weights *= np.array([synthetic_weight if r.get('synthetic') else 1. for r in rows])
    return weights / weights.mean()


def promotion_gate(current, candidate):
    reasons=[]
    if set(current)!=set(candidate):
        return dict(accepted=False,reasons=['Evaluation cohorts differ.'])
    gain=False
    for group, old in current.items():
        new=candidate[group]
        if new['support']!=old['support']:
            reasons.append(group+': support differs')
        if new['false_positive_rate']>.03:
            reasons.append(group+': empirical FPR exceeds 3%')
        if new['false_negatives']>old['false_negatives']:
            reasons.append(group+': recall regressed')
        gain |= new['false_negatives']<old['false_negatives']
    if not gain:
        reasons.append('No recall improvement on any test cohort.')
    return dict(accepted=not reasons,reasons=reasons)


def sms_only_bundle(active, candidate, gate, provenance):
    if not gate.get('accepted'):
        raise ValueError('SMS promotion gate rejected the candidate.')
    return dict(active, models=dict(active['models'],sms=candidate),
                metadata=dict(active['metadata'],sms_recall=provenance))
