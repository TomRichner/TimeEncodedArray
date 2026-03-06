"""
write_identical_tea_py.py — Write a deterministic discontinuous TEA file with appending.

The exact same operations are replicated in write_identical_tea_ml.m
so the resulting files can be compared byte-for-byte in MATLAB.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


def main():
    cv_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cross_validation_data')
    cv_dir = os.path.abspath(cv_dir)
    out = os.path.join(cv_dir, 'identical_py.mat')
    if os.path.exists(out):
        os.remove(out)

    SR = 500
    tea = TEA(out, SR, True, t_units='s', tea_version='1.0')

    # --- Write 1: initial chunk, 1000 samples, 3 channels ---
    # Continuous segment: samples 0-999 at 500 Hz
    t1 = np.arange(1000, dtype=np.float64) / SR
    s1 = np.column_stack([
        1.0 * np.ones(1000),
        2.0 * np.ones(1000),
        3.0 * np.ones(1000),
    ])
    tea.write(t1, s1)

    # --- Write 2: append with a 2-second GAP, 500 samples ---
    # Discontinuity: last sample of write 1 is t=1.998, first of write 2 is t=3.998
    t2 = t1[-1] + 2.0 + np.arange(1, 501, dtype=np.float64) / SR
    s2 = np.column_stack([
        4.0 * np.ones(500),
        5.0 * np.ones(500),
        6.0 * np.ones(500),
    ])
    tea.write(t2, s2)

    # --- Write 3: append continuous, 500 samples ---
    t3 = t2[-1] + np.arange(1, 501, dtype=np.float64) / SR
    s3 = np.column_stack([
        7.0 * np.ones(500),
        8.0 * np.ones(500),
        9.0 * np.ones(500),
    ])
    tea.write(t3, s3)

    print(f"Wrote: {out}")
    print(f"  N={tea.N}, C={tea.C}")
    print(f"  t range: [{t1[0]}, {t3[-1]}]")


if __name__ == '__main__':
    main()
