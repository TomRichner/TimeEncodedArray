"""
test_tea.py - Unit tests for the Python TEA class.

Run with: python -m pytest test_tea.py -v
"""

import numpy as np
import pytest
import tempfile
import shutil
import os
import sys

# Add parent dir to path so we can import tea
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tea import TEA


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix='tea_test_')
    yield d
    shutil.rmtree(d, ignore_errors=True)


def mat_path(temp_dir, name):
    return os.path.join(temp_dir, name + '.mat')


# ============ Test 1: Round-trip continuous ============

class TestRoundtripContinuous:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'cont')
        SR = 1000
        N = 5000
        t = np.arange(N) / SR
        samples = np.random.randn(N, 3)

        tea = TEA(f, SR, True, t_units='s')
        tea.write(t, samples)

        assert tea.N == N
        assert tea.C == 3

        Data, t_out, di = tea.read()
        np.testing.assert_allclose(t_out, t, atol=1e-12)
        np.testing.assert_allclose(Data, samples, atol=1e-12)
        assert not di['is_discontinuous']


# ============ Test 2: Discontinuous data ============

class TestDiscontinuous:
    def test_gap(self, temp_dir):
        f = mat_path(temp_dir, 'disc')
        SR = 1000
        t1 = np.arange(1000) / SR
        t2 = np.arange(1000, 2000) / SR + 2.0  # 2-second gap
        t = np.concatenate([t1, t2])
        samples = np.random.randn(2000, 1)

        tea = TEA(f, SR, True, t_units='s')
        tea.write(t, samples)

        _, _, di = tea.read()
        assert di['is_discontinuous']
        assert di['disc'].shape[0] == 1  # one gap


# ============ Test 3: Sample range read ============

class TestSampleRange:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'sr')
        SR = 1000
        N = 10000
        t = np.arange(N) / SR
        samples = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, samples)

        Data, t_out, _ = tea.read(s_range=(100, 499))
        assert Data.shape[0] == 400
        np.testing.assert_allclose(Data[0, 0], 101, atol=1e-12)  # 0-indexed: sample 100 has value 101


# ============ Test 4: Time range read ============

class TestTimeRange:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'tr')
        SR = 1000
        N = 10000
        t = np.arange(N) / SR
        samples = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, samples)

        Data, t_out, _ = tea.read(t_range=(1.0, 2.0))
        assert t_out[0] >= 1.0
        assert t_out[-1] <= 2.0
        assert len(t_out) > 0

    def test_at_discontinuity(self, temp_dir):
        """Query a time range that lands exactly at a discontinuity boundary."""
        f = mat_path(temp_dir, 'tr_disc')
        SR = 1000
        t1 = np.arange(1000) / SR      # 0.0 to 0.999
        t2 = 5.0 + np.arange(1000) / SR  # 5.0 to 5.999
        t = np.concatenate([t1, t2])
        s = np.arange(1, 2001, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, s)

        # Query only the first segment
        Data, t_out, di = tea.read(t_range=(0.0, 0.999))
        assert t_out[-1] <= 0.999 + 1e-10
        assert len(t_out) == 1000
        assert not di['is_discontinuous']

        # Query only the second segment
        Data2, t_out2, di2 = tea.read(t_range=(5.0, 5.999))
        assert t_out2[0] >= 5.0 - 1e-10
        assert len(t_out2) == 1000
        assert not di2['is_discontinuous']

    def test_spanning_gap(self, temp_dir):
        """Query a range that spans a discontinuity gap."""
        f = mat_path(temp_dir, 'tr_span')
        SR = 1000
        t1 = np.arange(1000) / SR
        t2 = 5.0 + np.arange(1000) / SR
        t = np.concatenate([t1, t2])
        s = np.arange(1, 2001, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, s)

        # Query from 0.5 to 5.5 — spans the gap
        Data, t_out, di = tea.read(t_range=(0.5, 5.5))
        assert t_out[0] >= 0.5
        assert t_out[-1] <= 5.5
        assert di['is_discontinuous']

    def test_beyond_file_range(self, temp_dir):
        """Query a time range entirely outside the file's data."""
        f = mat_path(temp_dir, 'tr_beyond')
        SR = 1000
        N = 1000
        t = np.arange(N) / SR  # 0 to 0.999
        s = np.ones((N, 1))

        tea = TEA(f, SR, True)
        tea.write(t, s)

        # Query beyond file end
        Data, t_out, _ = tea.read(t_range=(10.0, 20.0))
        assert len(t_out) == 0
        assert Data.shape[0] == 0

        # Query before file start
        Data2, t_out2, _ = tea.read(t_range=(-5.0, -1.0))
        assert len(t_out2) == 0

    def test_exact_boundaries(self, temp_dir):
        """Query exactly at the first and last sample times."""
        f = mat_path(temp_dir, 'tr_exact')
        SR = 1000
        N = 5000
        t = np.arange(N) / SR
        s = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, s)

        # Read entire range by exact bounds
        Data, t_out, _ = tea.read(t_range=(0.0, t[-1]))
        assert len(t_out) == N
        np.testing.assert_allclose(Data[0, 0], 1, atol=1e-12)
        np.testing.assert_allclose(Data[-1, 0], N, atol=1e-12)


# ============ Test 5: Channel selection ============

class TestChannelSelection:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'ch')
        SR = 100
        N = 500
        t = np.arange(N) / SR
        samples = np.column_stack([
            np.ones(N), 2 * np.ones(N), 3 * np.ones(N), 4 * np.ones(N)
        ])

        tea = TEA(f, SR, True)
        tea.write(t, samples)

        Data, _, _ = tea.read(channels=[2])
        np.testing.assert_allclose(Data[0, 0], 2, atol=1e-12)

    def test_with_ch_map(self, temp_dir):
        f = mat_path(temp_dir, 'ch2')
        SR = 100
        N = 500
        t = np.arange(N) / SR
        samples = np.column_stack([np.ones(N), 2 * np.ones(N)])

        tea = TEA(f, SR, True)
        tea.write(t, samples)
        # Write channels with explicit ch_map
        tea.write_channels(3 * np.ones((N, 1)), ch_map=[10])

        Data, _, _ = tea.read(channels=[10])
        np.testing.assert_allclose(Data[0, 0], 3, atol=1e-12)


# ============ Test 6: Append in time ============

class TestAppendTime:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'at')
        SR = 1000
        N = 5000
        t1 = np.arange(N) / SR
        s1 = np.ones((N, 2))

        tea = TEA(f, SR, True)
        tea.write(t1, s1)

        t2 = t1[-1] + np.arange(1, N + 1) / SR
        s2 = 2 * np.ones((N, 2))
        tea.write(t2, s2)

        assert tea.N == 2 * N
        Data, _, _ = tea.read(s_range=(N, N))
        np.testing.assert_allclose(Data[0, 0], 2, atol=1e-12)


# ============ Test 7: Append channels ============

class TestAppendChannels:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'ac')
        SR = 1000
        N = 5000
        t = np.arange(N) / SR
        s1 = np.ones((N, 2))

        tea = TEA(f, SR, True)
        tea.write(t, s1)

        tea.write_channels(3 * np.ones((N, 2)), ch_map=[3, 4])
        assert tea.C == 4

        Data, _, _ = tea.read(channels=[3])
        np.testing.assert_allclose(Data[0, 0], 3, atol=1e-12)


# ============ Test 8: Irregular data ============

class TestIrregular:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'irr')
        t = np.sort(np.random.rand(200) * 100)
        samples = np.random.randn(200, 2)

        tea = TEA(f, None, False)
        tea.write(t, samples)

        assert tea.N == 200
        Data, t_out, di = tea.read()
        np.testing.assert_allclose(t_out, t, atol=1e-12)
        assert not di['is_discontinuous']


# ============ Test 9: SR mismatch error ============

class TestSRMismatch:
    def test_error(self, temp_dir):
        f = mat_path(temp_dir, 'sm')
        SR = 1000
        t = np.arange(100) / SR

        tea = TEA(f, SR, True)
        tea.write(t, np.random.randn(100, 1))

        with pytest.raises(ValueError, match="SR mismatch"):
            TEA(f, 500, True)


# ============ Test 10: Monotonicity error ============

class TestMonotonicity:
    def test_error(self, temp_dir):
        f = mat_path(temp_dir, 'mono')
        t = np.array([0.0, 1.0, 0.5, 2.0])

        tea = TEA(f, 1000, True)
        with pytest.raises(ValueError, match="monotonically"):
            tea.write(t, np.random.randn(4, 1))


# ============ Test 11: Refresh ============

class TestRefresh:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'ref')
        SR = 1000
        N = 5000
        t = np.arange(N) / SR

        tea = TEA(f, SR, True)
        tea.write(t, np.random.randn(N, 2))

        # Delete dependents manually
        import h5py
        with h5py.File(f, 'r+') as hf:
            for var in ['t_coarse', 'df_t_coarse', 'isContinuous']:
                if var in hf:
                    del hf[var]

        # Refresh should recreate them
        tea.refresh()

        with h5py.File(f, 'r') as hf:
            assert 't_coarse' in hf
            assert 'df_t_coarse' in hf
            assert 'isContinuous' in hf


# ============ Test 12: Reopen existing file ============

class TestReopen:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'reop')
        SR = 1000
        N = 1000
        t = np.arange(N) / SR

        tea1 = TEA(f, SR, True, t_units='s')
        tea1.write(t, np.random.randn(N, 2))

        # Reopen
        tea2 = TEA(f, SR, True)
        assert tea2.N == N

        # Append via new handle
        t2 = t[-1] + np.arange(1, 501) / SR
        tea2.write(t2, np.random.randn(500, 2))
        assert tea2.N == 1500


# ============ Test 13: Info method ============

class TestInfo:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'inf')
        SR = 1000
        t = np.arange(1000) / SR

        tea = TEA(f, SR, True, t_units='s')
        tea.write(t, np.random.randn(1000, 3))

        s = tea.info()
        assert s['SR'] == 1000
        assert s['is_regular'] is True
        assert s['N'] == 1000
        assert s['C'] == 3
        assert s['t_units'] == 's'
        assert 't_first' in s
        assert 't_last' in s


# ============ Test 14: Append creates discontinuity ============

class TestAppendDiscontinuity:
    def test_junction_gap(self, temp_dir):
        f = mat_path(temp_dir, 'jdisc')
        SR = 1000
        t1 = np.arange(1000) / SR

        tea = TEA(f, SR, True)
        tea.write(t1, np.random.randn(1000, 2))

        # Append with gap
        t2 = t1[-1] + 2.0 + np.arange(1000) / SR  # 2-second gap
        tea.write(t2, np.random.randn(1000, 2))

        _, _, di = tea.read()
        assert di['is_discontinuous']
        assert di['disc'].shape[0] == 1


# ============ Test 15: Incremental matches full recompute ============

class TestIncrementalMatchesRefresh:
    def test_basic(self, temp_dir):
        f = mat_path(temp_dir, 'incr')
        SR = 500

        tea = TEA(f, SR, True)

        # Write initial chunk
        t1 = np.arange(2000) / SR
        tea.write(t1, np.random.randn(2000, 2))

        # Append with gap
        t2 = t1[-1] + 1.0 + np.arange(1000) / SR
        tea.write(t2, np.random.randn(1000, 2))

        # Append continuous
        t3 = t2[-1] + 1.0 / SR + np.arange(500) / SR
        tea.write(t3, np.random.randn(500, 2))

        # Capture incremental results
        import h5py
        with h5py.File(f, 'r') as hf:
            inc_isCont = bool(np.squeeze(hf['isContinuous'][()]))
            inc_disc = hf['disc'][()].T if 'disc' in hf else np.zeros((0, 2))
            inc_cont = hf['cont'][()].T if 'cont' in hf else np.zeros((0, 2))
            inc_tc = hf['t_coarse'][()].ravel()

        # Full recompute
        tea.refresh()

        with h5py.File(f, 'r') as hf:
            ref_isCont = bool(np.squeeze(hf['isContinuous'][()]))
            ref_disc = hf['disc'][()].T if 'disc' in hf else np.zeros((0, 2))
            ref_cont = hf['cont'][()].T if 'cont' in hf else np.zeros((0, 2))
            ref_tc = hf['t_coarse'][()].ravel()

        assert inc_isCont == ref_isCont
        np.testing.assert_array_equal(inc_disc, ref_disc)
        np.testing.assert_array_equal(inc_cont, ref_cont)
        np.testing.assert_allclose(inc_tc, ref_tc, atol=1e-12)
