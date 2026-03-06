"""
cross_validation_write_py.py — Write test TEA files from Python for MATLAB to read.

Creates deterministic test files with known data for cross-validation.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


def write_test_files(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- Case 1: Simple continuous ---
    f1 = os.path.join(out_dir, 'py_continuous.mat')
    SR = 1000
    N = 5000
    t = np.arange(N, dtype=np.float64) / SR
    # Deterministic data: channel k = k * ones
    samples = np.column_stack([np.ones(N), 2 * np.ones(N), 3 * np.ones(N)])

    tea = TEA(f1, SR, True, t_units='s')
    tea.write(t, samples)
    print(f"Wrote: {f1} (N={N}, C=3, continuous)")

    # --- Case 2: Discontinuous ---
    f2 = os.path.join(out_dir, 'py_discontinuous.mat')
    SR = 500
    t1 = np.arange(1000, dtype=np.float64) / SR
    t2 = 5.0 + np.arange(1000, dtype=np.float64) / SR  # 3-second gap
    t_disc = np.concatenate([t1, t2])
    s_disc = np.arange(1, 2001, dtype=np.float64).reshape(-1, 1)

    tea_disc = TEA(f2, SR, True, t_units='s')
    tea_disc.write(t_disc, s_disc)
    print(f"Wrote: {f2} (N=2000, C=1, discontinuous)")

    # --- Case 3: Appended ---
    f3 = os.path.join(out_dir, 'py_appended.mat')
    SR = 1000
    N1 = 3000
    t1 = np.arange(N1, dtype=np.float64) / SR
    s1 = 10 * np.ones((N1, 2))

    tea_app = TEA(f3, SR, True, t_units='s')
    tea_app.write(t1, s1)

    # Append continuous
    N2 = 2000
    t2 = t1[-1] + np.arange(1, N2 + 1, dtype=np.float64) / SR
    s2 = 20 * np.ones((N2, 2))
    tea_app.write(t2, s2)
    print(f"Wrote: {f3} (N={N1 + N2}, C=2, appended)")

    # --- Case 4: Irregular ---
    f4 = os.path.join(out_dir, 'py_irregular.mat')
    np.random.seed(42)
    t_irr = np.sort(np.random.rand(100) * 10)
    s_irr = np.arange(1, 101, dtype=np.float64).reshape(-1, 1)

    tea_irr = TEA(f4, None, False)
    tea_irr.write(t_irr, s_irr)
    print(f"Wrote: {f4} (N=100, C=1, irregular)")

    print(f"\nAll Python test files written to: {out_dir}")


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cross_validation_data')
    write_test_files(os.path.abspath(out_dir))
