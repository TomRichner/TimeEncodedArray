"""
cross_validation_read_ml.py — Read MATLAB-written TEA files and verify in Python.

Run after cross_validate_tea.m has written ml_*.mat files.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


def read_and_verify(cv_dir):
    n_pass = 0
    n_fail = 0

    print("Reading MATLAB-written TEA files in Python")
    print("=" * 60)

    # --- Case 1: ml_continuous.mat ---
    print("\nCase 1: ml_continuous.mat")
    try:
        f1 = os.path.join(cv_dir, 'ml_continuous.mat')
        tea = TEA(f1, 1000, True, t_units='s')

        assert tea.N == 4000, f"N: expected 4000, got {tea.N}"
        assert tea.C == 2, f"C: expected 2, got {tea.C}"

        Data, t_out, di = tea.read()
        assert Data.shape == (4000, 2), f"Shape: expected (4000,2), got {Data.shape}"
        assert abs(t_out[0] - 0.0) < 1e-10, f"First t: expected 0, got {t_out[0]}"
        assert abs(t_out[-1] - 3.999) < 1e-6, f"Last t: expected 3.999, got {t_out[-1]}"

        # Channel 1 = 5, Channel 2 = 6
        assert np.all(Data[:, 0] == 5), "Channel 1 should be all 5s"
        assert np.all(Data[:, 1] == 6), "Channel 2 should be all 6s"

        assert not di['is_discontinuous'], "Should be continuous"
        print("  PASS")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        n_fail += 1

    # --- Case 2: ml_discontinuous.mat ---
    print("\nCase 2: ml_discontinuous.mat")
    try:
        f2 = os.path.join(cv_dir, 'ml_discontinuous.mat')
        tea = TEA(f2, 500, True, t_units='s')

        assert tea.N == 1000, f"N: expected 1000, got {tea.N}"
        assert tea.C == 1, f"C: expected 1, got {tea.C}"

        Data, t_out, di = tea.read()

        # Data should be 1:1000
        expected = np.arange(1, 1001, dtype=np.float64)
        np.testing.assert_allclose(Data.ravel(), expected, atol=1e-10)

        # Gap between sample 499 and 500 (0-indexed)
        gap = t_out[500] - t_out[499]
        assert gap > 2.0, f"Should have gap > 2s, got {gap}"

        assert di['is_discontinuous'], "Should be discontinuous"
        assert di['disc'].shape[0] == 1, f"Expected 1 gap, got {di['disc'].shape[0]}"

        print("  PASS")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        n_fail += 1

    # --- Case 3: ml_channels.mat (with appended channel) ---
    print("\nCase 3: ml_channels.mat")
    try:
        f3 = os.path.join(cv_dir, 'ml_channels.mat')
        tea = TEA(f3, 1000, True)

        assert tea.N == 2000, f"N: expected 2000, got {tea.N}"
        assert tea.C == 3, f"C: expected 3, got {tea.C}"

        Data, t_out, _ = tea.read()

        # Channels 1-2 should be 7, channel 3 should be 8
        assert np.all(Data[:, 0] == 7), "Channel 1 should be all 7s"
        assert np.all(Data[:, 1] == 7), "Channel 2 should be all 7s"
        assert np.all(Data[:, 2] == 8), "Channel 3 should be all 8s"

        # Read by channel selection
        Data_ch3, _, _ = tea.read(channels=[3])
        assert Data_ch3.shape == (2000, 1), f"Shape: expected (2000,1), got {Data_ch3.shape}"
        assert np.all(Data_ch3 == 8), "Channel 3 should be all 8s"

        print("  PASS")
        n_pass += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        n_fail += 1

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {n_pass} passed, {n_fail} failed")
    if n_fail == 0:
        print("ALL CROSS-VALIDATION TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    return n_fail == 0


if __name__ == '__main__':
    cv_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cross_validation_data')
    cv_dir = os.path.abspath(cv_dir)
    success = read_and_verify(cv_dir)
    sys.exit(0 if success else 1)
