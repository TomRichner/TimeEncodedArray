"""
write_v2_test_cases.py - Write TEA schema-version-2 files for the MATLAB tests.

Usage (from the repo root):
    python python/tests/write_v2_test_cases.py [out_dir]

Default out_dir: cross_validation_data/v2. matlab/tests/test_tea_v2.m reads
these files (and runs this script through `uv run` when they are missing).
The sample values follow simple formulas so the MATLAB side can rebuild them.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tea
from tea import TEA, Stream

T_OFFSET = 1770000000


def ramp(n, c, scale=1.0):
    """(n, c) array 0, 1, 2, ... filled row by row (MATLAB: reshape(0:n*c-1, c, n)')."""
    return np.arange(n * c, dtype=np.float64).reshape(n, c) * scale


def two_rate_streams():
    t256 = np.arange(1024) / 256
    t1024 = np.concatenate([np.arange(1536) / 1024, 3.5 + np.arange(2560) / 1024])
    return [
        Stream(256, ['Fp1', 'Fp2', 'Cz'], 'uV', ramp(1024, 3), t256),
        Stream(1024, ['DC1', 'DC2'], ['mV', 'uV'], ramp(4096, 2, -0.5), t1024),
    ]


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tea.write(os.path.join(out_dir, 'v2_two_rates.mat'), two_rate_streams(),
              t_offset=T_OFFSET)
    tea.write(os.path.join(out_dir, 'v2_two_rates_compressed.mat'), two_rate_streams(),
              t_offset=T_OFFSET, compress=True)

    t200 = np.concatenate([np.arange(900) / 200, 10 + np.arange(600) / 200])
    tea.write(os.path.join(out_dir, 'v2_one_rate.mat'),
              [Stream(200, ['a', 'b', 'c', 'd'], 'uV', ramp(1500, 4), t200)],
              t_offset=T_OFFSET)

    v1 = os.path.join(out_dir, 'v1_python.mat')
    if os.path.exists(v1):
        os.remove(v1)
    TEA(v1, 500, True, t_units='s', hdr={'channel_names': ['A1', 'Bx2']}).write(
        np.arange(3000) / 500, ramp(3000, 2))

    print(f"TEA v2 test cases written to {out_dir}")


if __name__ == '__main__':
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, 'cross_validation_data', 'v2'))
