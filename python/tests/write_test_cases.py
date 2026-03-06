"""
write_test_cases.py — Write 8 deterministic TEA test files.

Each case uses hardcoded data identical to write_test_cases.m.
Output: cross_validation_data/py_case{1..8}.mat
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


def get_cv_dir():
    d = os.path.join(os.path.dirname(__file__), '..', '..', 'cross_validation_data')
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def clean(path):
    if os.path.exists(path):
        os.remove(path)


def case1(cv_dir):
    """Write 5000x3 continuous, SR=1000."""
    f = os.path.join(cv_dir, 'py_case1.mat')
    clean(f)
    SR = 1000
    t = np.arange(5000, dtype=np.float64) / SR
    s = np.column_stack([1.0 * np.ones(5000),
                         2.0 * np.ones(5000),
                         3.0 * np.ones(5000)])
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t, s)
    return f


def case2(cv_dir):
    """Write 2000x1 with 1 gap, SR=500."""
    f = os.path.join(cv_dir, 'py_case2.mat')
    clean(f)
    SR = 500
    t1 = np.arange(1000, dtype=np.float64) / SR
    t2 = 5.0 + np.arange(1000, dtype=np.float64) / SR
    t = np.concatenate([t1, t2])
    s = np.arange(1, 2001, dtype=np.float64).reshape(-1, 1)
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t, s)
    return f


def case3(cv_dir):
    """Write 1000x2, append 500x2 continuous, SR=1000."""
    f = os.path.join(cv_dir, 'py_case3.mat')
    clean(f)
    SR = 1000
    t1 = np.arange(1000, dtype=np.float64) / SR
    s1 = np.column_stack([10.0 * np.ones(1000), 20.0 * np.ones(1000)])
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t1, s1)

    t2 = t1[-1] + np.arange(1, 501, dtype=np.float64) / SR
    s2 = np.column_stack([30.0 * np.ones(500), 40.0 * np.ones(500)])
    tea.write(t2, s2)
    return f


def case4(cv_dir):
    """Write 1000x2, append 500x2 with 3-sec gap, SR=1000."""
    f = os.path.join(cv_dir, 'py_case4.mat')
    clean(f)
    SR = 1000
    t1 = np.arange(1000, dtype=np.float64) / SR
    s1 = np.column_stack([10.0 * np.ones(1000), 20.0 * np.ones(1000)])
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t1, s1)

    t2 = t1[-1] + 3.0 + np.arange(1, 501, dtype=np.float64) / SR
    s2 = np.column_stack([30.0 * np.ones(500), 40.0 * np.ones(500)])
    tea.write(t2, s2)
    return f


def case5(cv_dir):
    """Write 2000x2, write_channels 2000x1, SR=1000."""
    f = os.path.join(cv_dir, 'py_case5.mat')
    clean(f)
    SR = 1000
    t = np.arange(2000, dtype=np.float64) / SR
    s = np.column_stack([5.0 * np.ones(2000), 6.0 * np.ones(2000)])
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t, s)
    tea.write_channels(7.0 * np.ones((2000, 1)), ch_map=[3])
    return f


def case6(cv_dir):
    """Write 3000x1 with 3 gaps (4 segments of 750), SR=500."""
    f = os.path.join(cv_dir, 'py_case6.mat')
    clean(f)
    SR = 500
    segs = []
    for i in range(4):
        offset = i * (750.0 / SR + 2.0)  # 2-sec gap between segments
        segs.append(offset + np.arange(750, dtype=np.float64) / SR)
    t = np.concatenate(segs)
    s = np.arange(1, 3001, dtype=np.float64).reshape(-1, 1)
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t, s)
    return f


def case7(cv_dir):
    """Write 100x1 irregular (no SR)."""
    f = os.path.join(cv_dir, 'py_case7.mat')
    clean(f)
    # Deterministic non-uniform timestamps: t(k) = (k/99)^1.5 * 10
    t = ((np.arange(100, dtype=np.float64) / 99.0) ** 1.5) * 10.0
    s = np.arange(1, 101, dtype=np.float64).reshape(-1, 1)
    tea = TEA(f, None, False)
    tea.write(t, s)
    return f


def case8(cv_dir):
    """Write 1000x2 + append 500x2 with gap + append 500x2 continuous, SR=500."""
    f = os.path.join(cv_dir, 'py_case8.mat')
    clean(f)
    SR = 500
    t1 = np.arange(1000, dtype=np.float64) / SR
    s1 = np.column_stack([1.0 * np.ones(1000), 2.0 * np.ones(1000)])
    tea = TEA(f, SR, True, t_units='s')
    tea.write(t1, s1)

    # Append with 2-sec gap
    t2 = t1[-1] + 2.0 + np.arange(1, 501, dtype=np.float64) / SR
    s2 = np.column_stack([3.0 * np.ones(500), 4.0 * np.ones(500)])
    tea.write(t2, s2)

    # Append continuous
    t3 = t2[-1] + np.arange(1, 501, dtype=np.float64) / SR
    s3 = np.column_stack([5.0 * np.ones(500), 6.0 * np.ones(500)])
    tea.write(t3, s3)
    return f

def case9(cv_dir):
    """Write 1000x1 with t_offset, SR=1000."""
    f = os.path.join(cv_dir, 'py_case9.mat')
    clean(f)
    SR = 1000
    t = np.arange(1000, dtype=np.float64) / SR
    s = np.arange(1, 1001, dtype=np.float64).reshape(-1, 1)
    tea = TEA(f, SR, True, t_units='s',
              t_offset=np.int64(1770000000),
              t_offset_units='posix_s',
              t_offset_scale=1.0)
    tea.write(t, s)
    return f


def main():
    cv_dir = get_cv_dir()
    cases = [case1, case2, case3, case4, case5, case6, case7, case8, case9]
    for i, fn in enumerate(cases, 1):
        f = fn(cv_dir)
        print(f"  Case {i}: {os.path.basename(f)}")
    print(f"\nAll 9 Python test cases written to: {cv_dir}")


if __name__ == '__main__':
    main()
