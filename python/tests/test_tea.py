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
import warnings

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

        # Verify stored disc/cont values match MATLAB convention (1-indexed)
        import h5py
        with h5py.File(f, 'r') as hf:
            stored_disc = hf['disc'][()].T  # HDF5 transposed → (n, 2)
            stored_cont = hf['cont'][()].T
            stored_is_cont = bool(np.squeeze(hf['isContinuous'][()]))
        assert not stored_is_cont
        np.testing.assert_array_equal(stored_disc, [[1000, 1001]])
        np.testing.assert_array_equal(stored_cont, [[1, 1000], [1001, 2000]])


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


# ============ Test 17: Channel default (no ch_map) ============

class TestChannelDefault:
    def test_no_ch_map(self, temp_dir):
        """Select channels by default 1-indexed position (no explicit ch_map)."""
        f = mat_path(temp_dir, 'chd')
        SR = 100
        N = 500
        t = np.arange(N) / SR
        samples = np.column_stack([np.ones(N), 2 * np.ones(N)])

        tea = TEA(f, SR, True)
        tea.write(t, samples)

        Data, _, _ = tea.read(channels=[2])
        np.testing.assert_allclose(Data[0, 0], 2, atol=1e-12)


# ============ Test 18: Regularity validation error ============

class TestRegularityValidation:
    def test_error(self, temp_dir):
        """Highly irregular data with is_regular=True should raise."""
        f = mat_path(temp_dir, 'ie')
        SR = 1000
        t = np.cumsum([0, 0.001, 0.01, 0.1, 0.001, 0.5, 0.001, 0.3, 0.001, 0.2])
        samples = np.random.randn(len(t), 1)

        tea = TEA(f, SR, True)
        with pytest.raises(ValueError):
            tea.write(t, samples)


# ============ Test 19: Graceful fallback (no t_coarse) ============

class TestNoTcoarseFallback:
    def test_time_range_without_tcoarse(self, temp_dir):
        """Time-range read should work even if t_coarse is missing."""
        f = mat_path(temp_dir, 'nc')
        SR = 1000
        N = 5000
        t = np.arange(N, dtype=np.float64) / SR
        samples = np.arange(1, N + 1, dtype=np.float64).reshape(-1, 1)

        tea = TEA(f, SR, True)
        tea.write(t, samples)

        # Remove t_coarse and df_t_coarse
        import h5py
        with h5py.File(f, 'r+') as hf:
            if 't_coarse' in hf:
                del hf['t_coarse']
            if 'df_t_coarse' in hf:
                del hf['df_t_coarse']

        # Time-range read should still work via full-t fallback
        Data, t_out, _ = tea.read(t_range=(1.0, 2.0))
        assert t_out[0] >= 1.0
        assert t_out[-1] <= 2.0
        np.testing.assert_allclose(Data[0, 0], 1001, atol=1e-12)


# ============ Test 20: Append to already-discontinuous file ============

class TestAppendToDiscontinuous:
    def test_multi_gap_accumulation(self, temp_dir):
        """Three writes with gaps → 2 disc entries, 3 cont segments."""
        f = mat_path(temp_dir, 'adisc')
        SR = 1000
        tea = TEA(f, SR, True)

        t1 = np.arange(1000, dtype=np.float64) / SR
        tea.write(t1, np.random.randn(1000, 1))

        t2 = np.arange(5000, 6000, dtype=np.float64) / SR  # gap
        tea.write(t2, np.random.randn(1000, 1))

        t3 = np.arange(9000, 10000, dtype=np.float64) / SR  # gap
        tea.write(t3, np.random.randn(1000, 1))

        import h5py
        with h5py.File(f, 'r') as hf:
            stored_disc = hf['disc'][()].T
            stored_cont = hf['cont'][()].T
            stored_is_cont = bool(np.squeeze(hf['isContinuous'][()]))
        assert not stored_is_cont
        assert stored_disc.shape[0] == 2
        np.testing.assert_array_equal(stored_disc, [[1000, 1001], [2000, 2001]])
        np.testing.assert_array_equal(stored_cont, [[1, 1000], [1001, 2000], [2001, 3000]])


# ============ Test 21: Append chunk with internal gaps ============

class TestAppendInternalGaps:
    def test_gaps_within_chunk(self, temp_dir):
        """Append a single chunk that itself contains two internal gaps."""
        f = mat_path(temp_dir, 'igap')
        SR = 1000
        tea = TEA(f, SR, True)

        t1 = np.arange(500, dtype=np.float64) / SR
        tea.write(t1, np.random.randn(500, 1))

        # Append chunk with internal gaps
        seg_a = np.arange(500, 700, dtype=np.float64) / SR     # continuous with t1
        seg_b = np.arange(3000, 3200, dtype=np.float64) / SR   # gap
        seg_c = np.arange(6000, 6200, dtype=np.float64) / SR   # gap
        t2 = np.concatenate([seg_a, seg_b, seg_c])
        tea.write(t2, np.random.randn(len(t2), 1))

        import h5py
        with h5py.File(f, 'r') as hf:
            stored_disc = hf['disc'][()].T
            stored_is_cont = bool(np.squeeze(hf['isContinuous'][()]))
        assert not stored_is_cont
        assert stored_disc.shape[0] == 2
        np.testing.assert_array_equal(stored_disc, [[700, 701], [900, 901]])


# ============ Test 26: t_offset basic storage ============

class TestTOffsetBasic:
    def test_basic(self, temp_dir):
        f = os.path.join(temp_dir, 'toff.mat')
        SR = 1000
        t = np.arange(1000) / SR

        tea = TEA(f, SR, True, t_offset=np.int64(1770000000),
                  t_offset_units='posix_s', t_offset_scale=1.0)
        tea.write(t, np.random.randn(1000, 1))

        # Check stored in file
        import h5py
        with h5py.File(f, 'r') as hf:
            stored_offset = np.squeeze(hf['t_offset'][()])
            assert stored_offset == 1770000000

        # Check info returns it
        info = tea.info()
        assert info['t_offset'] == np.int64(1770000000)
        assert info['t_offset_units'] == 'posix_s'
        assert info['t_offset_scale'] == 1.0


# ============ Test 27: write_absolute basic ============

class TestWriteAbsolute:
    def test_basic(self, temp_dir):
        f = os.path.join(temp_dir, 'wabs.mat')
        SR = 1000
        t_offset = np.int64(1000)

        tea = TEA(f, SR, True, t_offset=t_offset,
                  t_offset_units='s', t_offset_scale=1.0)

        # Absolute timestamps
        t_abs = 1000.0 + np.arange(1000) / SR
        tea.write_absolute(t_abs, np.random.randn(1000, 1))

        # Should be stored as relative
        import h5py
        with h5py.File(f, 'r') as hf:
            t_stored = hf['t'][0, :]
        np.testing.assert_allclose(t_stored[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(t_stored[-1], 0.999, atol=1e-10)


# ============ Test 28: write_absolute cross-unit ============

class TestWriteAbsoluteCrossUnit:
    def test_cross_unit(self, temp_dir):
        f = os.path.join(temp_dir, 'wabx.mat')
        SR = 0.03  # samples/us (30 kHz)
        t_offset = np.int64(1000)  # seconds

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tea = TEA(f, SR, True, t_units='us',
                      t_offset=t_offset, t_offset_units='posix_s',
                      t_offset_scale=1e6)

        # Absolute timestamps in microseconds (1e9 us = 1000 s)
        t_abs_us = 1e9 + np.arange(100) * (1 / SR)
        tea.write_absolute(t_abs_us, np.random.randn(100, 1))

        # Stored t should be relative
        import h5py
        with h5py.File(f, 'r') as hf:
            t_stored = hf['t'][0, :]
        np.testing.assert_allclose(t_stored[0], 0.0, atol=1.0)


# ============ Test 29: write_absolute without t_offset errors ============

class TestWriteAbsoluteNoOffset:
    def test_no_offset(self, temp_dir):
        f = os.path.join(temp_dir, 'wano.mat')
        tea = TEA(f, 1000, True)
        with pytest.raises(ValueError, match="t_offset"):
            tea.write_absolute(np.arange(10, dtype=float), np.random.randn(10, 1))


# ============ Test 30: SR units warning ============

class TestSRUnitsWarning:
    def test_warning(self, temp_dir):
        f = os.path.join(temp_dir, 'sruw.mat')
        with pytest.warns(UserWarning, match="samples/us"):
            TEA(f, 0.03, True, t_units='us')


# ============ Test 31: precision warning ============

class TestPrecisionWarning:
    def test_warning(self, temp_dir):
        f = os.path.join(temp_dir, 'prec.mat')
        SR = 1  # 1 sample/us
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tea = TEA(f, SR, True, t_units='us')

        t_abs = 1.77e15 + np.arange(100, dtype=float)
        with pytest.warns(UserWarning, match="precision"):
            tea.write(t_abs, np.random.randn(100, 1))


# ============ Test 32: Compression ============

class TestCompression:
    def test_roundtrip(self, temp_dir):
        """Write with compress=True, verify data and HDF5 filter."""
        f = os.path.join(temp_dir, 'comp.mat')
        SR = 1000
        N = 5000
        t = np.arange(N) / SR
        samples = np.random.randn(N, 3)

        tea = TEA(f, SR, True, compress=True)
        tea.write(t, samples)

        # Read back and verify data
        Data, t_out, di = tea.read()
        np.testing.assert_allclose(t_out, t, atol=1e-12)
        np.testing.assert_allclose(Data, samples, atol=1e-12)
        assert not di['is_discontinuous']

        # Verify HDF5 datasets have gzip compression
        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['t'].compression == 'gzip'
            assert hf['t'].compression_opts == 1
            assert hf['t'].shuffle is True
            assert hf['Samples'].compression == 'gzip'
            assert hf['Samples'].compression_opts == 1
            assert hf['Samples'].shuffle is True

    def test_append_preserves_compression(self, temp_dir):
        """Appended data should also be compressed."""
        f = os.path.join(temp_dir, 'comp_app.mat')
        SR = 1000
        N = 2000

        tea = TEA(f, SR, True, compress=True)
        t1 = np.arange(N) / SR
        tea.write(t1, np.random.randn(N, 2))

        t2 = t1[-1] + np.arange(1, N + 1) / SR
        tea.write(t2, np.random.randn(N, 2))

        assert tea.N == 2 * N

        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['Samples'].compression == 'gzip'

    def test_default_no_compression(self, temp_dir):
        """Default (compress=False) should have no compression."""
        f = os.path.join(temp_dir, 'nocomp.mat')
        SR = 1000
        t = np.arange(1000) / SR

        tea = TEA(f, SR, True)
        tea.write(t, np.random.randn(1000, 2))

        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['Samples'].compression is None


# ============ Test: Pre-allocation ============

class TestPreallocation:
    def test_preallocate_roundtrip(self, temp_dir):
        """Write once within budget, finalize, read back."""
        f = mat_path(temp_dir, 'prealloc_rt')
        SR = 1000
        N = 3000
        C = 4
        t = np.arange(N) / SR
        samples = np.random.randn(N, C)

        tea = TEA(f, SR, True, t_units='s', expected_length=10000, expected_channels=C)
        tea.write(t, samples)

        # Before finalize: logical N should be correct
        assert tea.N == N
        assert tea.C == C

        tea.finalize()

        assert tea.N == N
        assert tea.C == C

        Data, t_out, di = tea.read()
        np.testing.assert_allclose(t_out, t, atol=1e-12)
        np.testing.assert_allclose(Data, samples, atol=1e-12)

    def test_preallocate_multiple_writes(self, temp_dir):
        """Stream multiple writes within budget, finalize, verify."""
        f = mat_path(temp_dir, 'prealloc_multi')
        SR = 500
        C = 3
        chunk_size = 1000

        tea = TEA(f, SR, True, t_units='s', expected_length=5000)

        all_t = []
        all_samples = []
        for i in range(4):  # 4 x 1000 = 4000 < 5000
            t = (i * chunk_size + np.arange(chunk_size)) / SR
            s = np.random.randn(chunk_size, C)
            tea.write(t, s)
            all_t.append(t)
            all_samples.append(s)

        assert tea.N == 4000
        tea.finalize()
        assert tea.N == 4000

        Data, t_out, _ = tea.read()
        np.testing.assert_allclose(t_out, np.concatenate(all_t), atol=1e-12)
        np.testing.assert_allclose(Data, np.vstack(all_samples), atol=1e-12)

        # Verify physical size matches logical after finalize
        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['t'].shape == (1, 4000)
            assert hf['Samples'].shape == (C, 4000)

    def test_preallocate_exceeds(self, temp_dir):
        """Write past expected_length — verify auto-resize works."""
        f = mat_path(temp_dir, 'prealloc_exceed')
        SR = 1000
        C = 2

        tea = TEA(f, SR, True, t_units='s', expected_length=500)

        # Write 500, then 500 more (total 1000 > 500 allocation)
        t1 = np.arange(500) / SR
        tea.write(t1, np.random.randn(500, C))

        t2 = (500 + np.arange(500)) / SR
        s2 = np.random.randn(500, C)
        tea.write(t2, s2)  # should auto-resize

        assert tea.N == 1000
        tea.finalize()

        Data, t_out, _ = tea.read()
        assert len(t_out) == 1000
        np.testing.assert_allclose(Data[500:, :], s2, atol=1e-12)

    def test_preallocate_channels(self, temp_dir):
        """Pre-allocate channels, add via write_channels(), finalize."""
        f = mat_path(temp_dir, 'prealloc_ch')
        SR = 1000
        N = 2000
        t = np.arange(N) / SR
        initial_samples = np.random.randn(N, 2)

        tea = TEA(f, SR, True, t_units='s', expected_channels=10)
        tea.write(t, initial_samples)

        assert tea.C == 2

        # Add 3 more channels
        extra = np.random.randn(N, 3)
        tea.write_channels(extra)

        assert tea.C == 5

        tea.finalize()

        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['Samples'].shape == (5, N)

        Data, _, _ = tea.read()
        np.testing.assert_allclose(Data[:, :2], initial_samples, atol=1e-12)
        np.testing.assert_allclose(Data[:, 2:5], extra, atol=1e-12)

    def test_default_no_preallocation(self, temp_dir):
        """Default behavior unchanged when no expected_length/channels given."""
        f = mat_path(temp_dir, 'no_prealloc')
        SR = 1000
        N = 1000
        C = 2
        t = np.arange(N) / SR
        samples = np.random.randn(N, C)

        tea = TEA(f, SR, True, t_units='s')
        tea.write(t, samples)

        assert tea.N == N
        assert tea.C == C

        # finalize is safe to call even without pre-allocation
        tea.finalize()

        import h5py
        with h5py.File(f, 'r') as hf:
            assert hf['t'].shape == (1, N)
            assert hf['Samples'].shape == (C, N)

        Data, t_out, _ = tea.read()
        np.testing.assert_allclose(t_out, t, atol=1e-12)
        np.testing.assert_allclose(Data, samples, atol=1e-12)
