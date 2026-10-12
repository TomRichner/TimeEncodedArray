"""
test_tea_v2.py - Tests for TEA schema version 2 (several sample rates per file)
and for reading version-1 files through the version-2 reader.

Run with: python -m pytest python/tests -q
"""

import os
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tea
from tea import TEA, Stream

T_OFFSET = 1770000000  # POSIX seconds


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / 'clip.mat')


def regular_t(rate, n, start=0.0):
    return start + np.arange(n) / rate


def two_rate_streams():
    """256 Hz, 3 channels, 4 s; 1024 Hz, 2 channels, 4 s with a 2 s gap after 1.5 s."""
    t256 = regular_t(256, 1024)
    x256 = np.arange(1024 * 3, dtype=np.float64).reshape(1024, 3)
    t1024 = np.concatenate([regular_t(1024, 1536), regular_t(1024, 2560, start=3.5)])
    x1024 = -np.arange(len(t1024) * 2, dtype=np.float64).reshape(-1, 2) * 0.5
    return [
        Stream(256, ['Fp1', 'Fp2', 'Cz'], 'uV', x256, t256),
        Stream(1024, ['DC1', 'DC2'], ['mV', 'uV'], x1024, t1024),
    ]


# ============ Version-1 files still read ============

class TestV1Files:
    def test_read_v1_through_v2_reader(self, path):
        t = regular_t(500, 3000)
        x = np.random.randn(3000, 2)
        TEA(path, 500, True, t_units='s', t_offset=np.int64(T_OFFSET),
            t_offset_units='posix_s', hdr={'channel_names': ['A1', 'Bx2']}).write(t, x)

        f = tea.read(path)
        assert f.schema_version == 1
        assert f.rates == [500.0]
        assert f.t_offset == T_OFFSET
        s = f.stream(500)
        assert s.channel_names == ['A1', 'Bx2']
        assert s.units == ['', '']
        data, t_out, _ = s.read()
        np.testing.assert_array_equal(data, x)
        np.testing.assert_array_equal(t_out, t)
        assert f.stream() is s and s.group == '/'

    def test_v1_default_names_and_tea_api(self, path):
        t = regular_t(100, 700)
        x = np.random.randn(700, 3)
        t1 = TEA(path, 100, True)
        t1.write(t, x)
        assert t1.schema_version == 1
        assert t1.rates == [100.0]
        s = t1.stream(100)
        assert s.channel_names == ['Ch1', 'Ch2', 'Ch3']
        data, _, _ = s.read(channels=['Ch3', 'Ch1'], s_range=(10, 19))
        np.testing.assert_array_equal(data, x[10:20, [2, 0]])

    def test_v1_irregular(self, path):
        t = np.sort(np.random.rand(200)) * 100
        x = np.random.randn(200, 1)
        TEA(path, None, False).write(t, x)
        f = tea.read(path)
        assert f.rates == [None]
        data, t_out, _ = f.stream().read()
        np.testing.assert_array_equal(t_out, t)


# ============ Version-2 round trips ============

class TestV2RoundTrip:
    def test_two_rates(self, path):
        streams = two_rate_streams()
        tea.write(path, streams, t_offset=T_OFFSET)

        f = tea.read(path)
        assert f.schema_version == 2
        assert f.rates == [256.0, 1024.0]
        assert f.t_offset == T_OFFSET
        assert f.t_offset_units == 'posix_s'
        assert f.t_units == 's'
        assert f.channel_names == ['Fp1', 'Fp2', 'Cz', 'DC1', 'DC2']
        assert f.channel_rates == [256.0] * 3 + [1024.0] * 2
        for s_in in streams:
            s = f.stream(s_in.rate)
            assert s.rate == s_in.rate
            assert s.channel_names == list(s_in.channel_names)
            data, t_out, _ = s.read()
            np.testing.assert_array_equal(data, s_in.samples)
            np.testing.assert_array_equal(t_out, s_in.t)
        assert f.stream(1024).units == ['mV', 'uV']
        assert f.stream(256).units == ['uV'] * 3
        assert f.stream(256).N == 1024 and f.stream(256).C == 3

        # Stored layout
        with h5py.File(path, 'r') as h:
            assert float(np.squeeze(h['schema_version'][()])) == 2
            assert set(h['rates'].keys()) == {'r1', 'r2'}
            assert h['rates/r1/Samples'].shape == (3, 1024)  # MATLAB [N x C]
            assert h['rates/r2/t'].shape == (1, 4096)
            # Several rates: no version-1 fields at the top level
            assert 't' not in h and 'Samples' not in h and 'SR' not in h
            stored_rates = h['channel_rates'][()].ravel()
            np.testing.assert_array_equal(stored_rates, [256, 256, 256, 1024, 1024])

        # MATLAB v7.3 header
        with open(path, 'rb') as fid:
            assert fid.read(10) == b'MATLAB 7.3'

    def test_tea_class_on_multi_rate_file(self, path):
        tea.write(path, two_rate_streams(), t_offset=T_OFFSET)
        handle = TEA(path, 256, True)
        assert handle.schema_version == 2
        assert handle.rates == [256.0, 1024.0]
        data, _, _ = handle.stream(1024).read(s_range=(0, 9))
        assert data.shape == (10, 2)
        with pytest.raises(ValueError, match='several sample rates'):
            handle.read()
        with pytest.raises(RuntimeError):
            handle.write(np.arange(3) / 256 + 100, np.zeros((3, 3)))

    def test_gap_inside_one_rate(self, path):
        streams = two_rate_streams()
        tea.write(path, streams)
        s = tea.read(path).stream(1024)
        data, t_out, di = s.read()
        np.testing.assert_array_equal(t_out, streams[1].t)
        assert di['is_discontinuous']
        np.testing.assert_array_equal(di['disc'], [[1535, 1536]])
        assert s.info()['isContinuous'] is False
        with h5py.File(path, 'r') as h:
            np.testing.assert_array_equal(h['rates/r2/disc'][()].T, [[1536, 1537]])
            np.testing.assert_array_equal(h['rates/r2/cont'][()].T, [[1, 1536], [1537, 4096]])
            assert not bool(np.squeeze(h['rates/r2/isContinuous'][()]))
            assert bool(np.squeeze(h['rates/r1/isContinuous'][()]))
            assert 'disc' not in h['rates/r1']
        # A time window inside the gap is empty; one spanning it has both sides
        d, t_gap, _ = s.read(t_range=(2.0, 3.0))
        assert len(t_gap) == 0
        d, t_span, di = s.read(t_range=(1.4, 3.6))
        assert t_span[0] >= 1.4 and t_span[-1] <= 3.6
        assert di['is_discontinuous']

    def test_one_rate_reads_like_v1(self, path):
        t = np.concatenate([regular_t(200, 900), regular_t(200, 600, start=10.0)])
        x = np.random.randn(len(t), 4)
        tea.write(path, [Stream(200, ['a', 'b', 'c', 'd'], 'uV', x, t)], t_offset=T_OFFSET)

        # Version-1 API, unchanged
        v1 = TEA(path, 200, True)
        assert v1.N == len(t) and v1.C == 4
        data, t_out, di = v1.read()
        np.testing.assert_array_equal(data, x)
        np.testing.assert_array_equal(t_out, t)
        assert di['is_discontinuous']
        data, t_out, _ = v1.read(channels=[2], t_range=(10.0, 10.5))
        np.testing.assert_array_equal(data[:, 0], x[(t >= 10.0) & (t <= 10.5), 1])
        info = v1.info()
        assert info['isContinuous'] is False

        with h5py.File(path, 'r') as h:
            # Top-level version-1 fields are links to the stream's datasets
            for name in ('t', 'Samples', 'SR', 'isRegular', 't_coarse', 'df_t_coarse',
                         'isContinuous', 'cont', 'disc'):
                assert h[name] == h['rates/r1/' + name], name
            np.testing.assert_array_equal(h['t_offset'][()].ravel(), [T_OFFSET])

        # Data stored once: the file is not twice the size of the samples. The
        # last chunk of each dataset is stored at full size (uncompressed), so
        # allow one Samples chunk and one t chunk of padding.
        padding = 8 * (4 * tea.chunk_samples(4) + tea.chunk_samples(1))
        assert os.path.getsize(path) < 1.5 * x.nbytes + padding + 100_000

        with pytest.raises(RuntimeError, match='version-2'):
            v1.write(t[-1] + 1 + np.arange(3) / 200, np.zeros((3, 4)))

    def test_irregular_stream(self, path):
        t = np.cumsum(np.random.rand(300) + 0.01)
        x = np.random.randn(300, 1)
        tea.write(path, [Stream(100, ['ev'], 'au', x, t, is_regular=False),
                         Stream(10, ['slow'], 'au', np.ones((20, 1)), regular_t(10, 20))])
        s = tea.read(path).stream(100)
        assert s.is_regular is False
        data, t_out, di = s.read()
        np.testing.assert_array_equal(t_out, t)
        assert not di['is_discontinuous']


# ============ Chunks, compression, partial reads ============

class TestChunkedReads:
    def test_appended_blocks_and_partial_reads(self, path):
        rate_a, rate_b = 512.0, 2048.0
        n_a, n_b = 5 * 512, 5 * 2048
        t_a, t_b = regular_t(rate_a, n_a), regular_t(rate_b, n_b)
        x_a = np.random.randn(n_a, 6)
        x_b = np.random.randn(n_b, 2)
        with tea.Writer(path, t_offset=T_OFFSET, compress=True) as w:
            w.add_stream(rate_a, [f'E{i}' for i in range(6)], 'uV')
            w.add_stream(rate_b, ['M1', 'M2'], 'uV')
            # Blocks of odd sizes, interleaved between streams
            for lo in range(0, n_a, 333):
                w.append(rate_a, t_a[lo:lo + 333], x_a[lo:lo + 333])
            for lo in range(0, n_b, 1777):
                w.append(rate_b, t_b[lo:lo + 1777], x_b[lo:lo + 1777])

        with h5py.File(path, 'r') as h:
            for k, C in (('r1', 6), ('r2', 2)):
                ds = h[f'rates/{k}/Samples']
                assert ds.chunks == (C, tea.chunk_samples(C))
                assert ds.compression == 'gzip' and ds.compression_opts == 1
                assert ds.shuffle is False
                assert h[f'rates/{k}/t'].chunks == (1, tea.chunk_samples(1))

        f = tea.read(path)
        a, b = f.stream(rate_a), f.stream(rate_b)
        # Sample range spanning several appended blocks (333 samples each)
        data, t_out, _ = a.read(s_range=(480, 1020))
        np.testing.assert_array_equal(data, x_a[480:1021])
        np.testing.assert_array_equal(t_out, t_a[480:1021])
        # Time range, channel subset by name and number, out of order
        data, t_out, _ = b.read(channels=['M2', 1], t_range=(1.0, 1.25))
        sel = (t_b >= 1.0) & (t_b <= 1.25)
        np.testing.assert_array_equal(t_out, t_b[sel])
        np.testing.assert_array_equal(data, x_b[sel][:, [1, 0]])
        # The t_coarse index matches a full recompute
        with h5py.File(path, 'r') as h:
            np.testing.assert_array_equal(h['rates/r1/t_coarse'][()].ravel(), t_a[::512])
            np.testing.assert_array_equal(h['rates/r2/t_coarse'][()].ravel(), t_b[::2048])

    def test_uncompressed_by_default(self, path):
        tea.write(path, two_rate_streams())
        with h5py.File(path, 'r') as h:
            assert h['rates/r1/Samples'].compression is None

    def test_chunk_samples_targets_512_kb_per_chunk(self):
        assert tea.chunk_samples(276) == 237            # 276 x 8 x 237 = 523 KB
        assert tea.chunk_samples(276, np.int32) == 474
        assert tea.chunk_samples(276, np.float32) == 474
        assert tea.chunk_samples(46) == 1424
        assert tea.chunk_samples(1) == tea.CHUNK_SAMPLES_MAX
        assert tea.chunk_samples(100_000) == tea.CHUNK_SAMPLES_MIN
        for c in (1, 4, 46, 276, 1024):
            assert c * 8 * tea.chunk_samples(c) <= 1024 * 1024 or c * 8 * tea.CHUNK_SAMPLES_MIN > 1024 * 1024

    def test_wide_stream_reads_across_chunk_boundaries(self, path):
        # 300 channels: 218 samples per chunk, so a 2000-sample stream has
        # nine chunks and the reads below cross several boundaries.
        rate, n, C = 1000.0, 2000, 300
        t = regular_t(rate, n)
        x = np.random.randn(n, C)
        with tea.Writer(path, t_offset=T_OFFSET, compress=True) as w:
            w.add_stream(rate, [f'E{i}' for i in range(C)], 'uV')
            for lo in range(0, n, 700):
                w.append(rate, t[lo:lo + 700], x[lo:lo + 700])
        with h5py.File(path, 'r') as h:
            assert h['rates/r1/Samples'].chunks == (C, 218)
            assert h['rates/r1/t'].chunks == (1, 8192)
        s = tea.read(path).stream(rate)
        data, t_out, _ = s.read(s_range=(200, 1300))
        np.testing.assert_array_equal(data, x[200:1301])
        np.testing.assert_array_equal(t_out, t[200:1301])
        data, t_out, _ = s.read(channels=['E299', 'E0'], t_range=(0.43, 0.44))
        sel = (t >= 0.43) & (t <= 0.44)
        np.testing.assert_array_equal(data, x[sel][:, [299, 0]])
        # The schema-1 class follows the same rule, capped at the allocation
        p1 = path.replace('.mat', '_v1.mat')
        tea.TEA(p1, rate, True, t_units='s').write(t, x)
        with h5py.File(p1, 'r') as h:
            assert h['Samples'].chunks == (C, 218)
            assert h['t'].chunks == (1, min(8192, n))
        d1, t1, _ = tea.TEA(p1, rate, True).read(s_range=(210, 1290))
        np.testing.assert_array_equal(d1, x[210:1291])


# ============ Writer validation ============

class TestWriterErrors:
    def test_duplicate_rate(self, path):
        with pytest.raises(ValueError, match='already exists'):
            with tea.Writer(path) as w:
                w.add_stream(256, ['a'], 'uV')
                w.add_stream(256.0, ['b'], 'uV')

    def test_shape_and_order(self, path):
        with pytest.raises(ValueError):
            with tea.Writer(path) as w:
                w.add_stream(100, ['a', 'b'], 'uV')
                w.append(100, regular_t(100, 10), np.zeros((10, 3)))
        with pytest.raises(ValueError, match='increasing'):
            with tea.Writer(path) as w:
                w.add_stream(100, ['a'], 'uV')
                w.append(100, regular_t(100, 10), np.zeros(10))
                w.append(100, regular_t(100, 10), np.zeros(10))

    def test_unknown_rate_and_empty_stream(self, path):
        with pytest.raises(KeyError):
            with tea.Writer(path) as w:
                w.add_stream(100, ['a'], 'uV')
                w.append(200, [0.0], [1.0])
        with pytest.raises(ValueError, match='no samples'):
            tea.write(path, [Stream(100, ['a'], 'uV', np.zeros((0, 1)), [])])

    def test_too_fast_for_rate(self, path):
        with pytest.raises(ValueError, match='0.9'):
            tea.write(path, [Stream(100, ['a'], 'uV', np.zeros((10, 1)), regular_t(1000, 10))])

    def test_stream_is_read_only(self, path):
        tea.write(path, two_rate_streams())
        with pytest.raises(RuntimeError, match='read-only'):
            tea.read(path).stream(256).write(np.arange(3.0) + 10, np.zeros((3, 3)))
