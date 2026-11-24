#!/usr/bin/env python3
"""Prepare available licensed local raw sources; no network access."""
import os
import sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spamdet.curation import prepare
if __name__ == '__main__':
    prepare()
