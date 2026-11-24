"""Staged precision refinement: snapshot -> freeze -> train -> evaluate/install."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spamdet.refinement import snapshot, freeze, train_candidates, evaluate_and_install


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage', choices=['snapshot','freeze','train','evaluate'])
    stage=parser.parse_args().stage
    result={'snapshot':'snapshot verified'} if stage=='snapshot' else None
    if stage=='snapshot':
        snapshot()
    elif stage=='freeze':
        result=freeze()
        result={k:v for k,v in result.items() if k!='new_test'}
    elif stage=='train':
        result=train_candidates()
    else:
        report=evaluate_and_install()
        result={k:report[k] for k in ['selected','new_holdout','by_channel_language','artifact_sha256']}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
