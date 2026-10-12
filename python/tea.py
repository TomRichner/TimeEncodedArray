"""
tea.py - Python implementation of the TEA (Time-Encoded Array) format.

Produces MATLAB v7.3 (.mat) compatible HDF5 files that can be opened
by the MATLAB TEA class or any HDF5 reader.

Two layers:
- `TEA`: one file, one sample rate (schema version 1). Unchanged API.
- `write` / `Writer` / `read`: schema version 2, several sample rates in one
  file, one group per rate under `/rates`. A version-2 file with one rate also
  carries the version-1 fields, so `TEA` and version-1 readers open it as before.

Uses h5py for all HDF5 operations; the MATLAB-side helpers (datasets with
`MATLAB_class` attributes, the v7.3 userblock header) live in `tea_h5`.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import numpy.typing as npt

from tea_h5 import (read_char_matrix, read_scalar, read_string, schema_version_of,
                    write_array, write_char_matrix, write_hdr_fields,
                    write_matlab_header, write_scalar, write_string)

logger = logging.getLogger(__name__)

# Type aliases
ArrayFloat64 = npt.NDArray[np.float64]

# HDF5 chunking: a chunk spans all channels and about CHUNK_BYTES of samples.
# Chunks of this size keep partial reads cheap (a one-second window from a
# compressed file decompresses a megabyte or less, not a 70 MB chunk; measured
# 30x faster than 32000-sample chunks at 276 channels, same file size) while
# staying under HDF5's 1 MB default chunk cache. The sample count is derived
# from the channel count and dtype, bounded so that a one-channel stream does
# not get thousands of tiny chunks and a very wide one does not exceed the
# cache. The MATLAB class uses the same rule. Cost: HDF5 stores the last,
# partial chunk of a dataset at full size, so an uncompressed file carries up
# to CHUNK_BYTES (Samples) plus 64 KB (t) of padding; it compresses away.
CHUNK_BYTES = 512 * 1024
CHUNK_SAMPLES_MIN = 64
CHUNK_SAMPLES_MAX = 8192


def chunk_samples(n_channels: int, dtype=np.float64, target_bytes: int = CHUNK_BYTES) -> int:
    """Samples per HDF5 chunk for `n_channels` of `dtype`, about `target_bytes` per chunk.

    276 float64 channels give 237 samples (about 512 KB); 46 give 1424; one
    channel gives the 8192 cap (64 KB).
    """
    itemsize = np.dtype(dtype).itemsize
    per_sample = max(1, int(n_channels)) * itemsize
    return int(min(CHUNK_SAMPLES_MAX, max(CHUNK_SAMPLES_MIN, target_bytes // per_sample)))


# Highest schema version this module writes and reads.
SCHEMA_VERSION = 2

# Group holding one subgroup per sample rate in a version-2 file.
RATES_GROUP = 'rates'

# Datasets of a stream group that a one-rate version-2 file also links at the
# top level, so version-1 readers find them where they expect them.
_V1_LINKED = ('t', 'Samples', 'SR', 'isRegular', 't_coarse', 'df_t_coarse',
              'isContinuous', 'cont', 'disc', 'scale', 'offset')

# Storage classes the writers accept for Samples, with their MATLAB class names.
STORAGE_CLASSES = {
    np.dtype(np.float64): 'double',
    np.dtype(np.float32): 'single',
    np.dtype(np.int16): 'int16',
    np.dtype(np.int32): 'int32',
    np.dtype(np.int64): 'int64',
}
_DTYPE_ALIASES = {'double': np.float64, 'single': np.float32}


# =====================================================================
# Storage class, scale and offset
# =====================================================================


def compression_kwargs(compress: bool, dtype=np.float64) -> dict:
    """h5py filter arguments for a `t` or `Samples` dataset.

    DEFLATE level 1 when `compress`. The byte-shuffle filter is used for the
    integer classes only: on int32 EEG counts it gave files 25-28 % smaller
    and faster writes and reads (276 channels, 492 s: 103 MB vs 138 MB, read
    0.9 s vs 1.6 s), while on float64 physical values it hides the repeated
    8-byte patterns DEFLATE finds (438 MB vs 273 MB on the same clip).
    """
    if not compress:
        return {}
    kw = {'compression': 'gzip', 'compression_opts': 1}
    if np.issubdtype(np.dtype(dtype), np.integer):
        kw['shuffle'] = True
    return kw


def storage_dtype(dtype) -> np.dtype:
    """The numpy dtype of a storage class (`np.int32`, `'int32'`, `'single'`, ...)."""
    if isinstance(dtype, str) and dtype in _DTYPE_ALIASES:
        dtype = _DTYPE_ALIASES[dtype]
    dt = np.dtype(dtype)
    if dt not in STORAGE_CLASSES:
        raise ValueError(f"dtype {dt} is not a TEA storage class; use one of "
                         f"{', '.join(v for v in STORAGE_CLASSES.values())}")
    return dt


def _scaling(dtype: np.dtype, scale, offset, C: int
             ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`scale` and `offset` as (C,) float64 arrays, or (None, None) when unscaled.

    Only an integer storage class may carry them. `offset` without `scale`
    means a scale of 1; `scale` without `offset` means an offset of 0.
    """
    if scale is None and offset is None:
        return None, None
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"scale and offset need an integer dtype, not {dtype}")
    scale_arr = np.broadcast_to(np.asarray(1.0 if scale is None else scale, dtype=np.float64),
                                (C,)).copy()
    offset_arr = np.broadcast_to(np.asarray(0.0 if offset is None else offset, dtype=np.float64),
                                 (C,)).copy()
    if not np.all(np.isfinite(scale_arr)) or np.any(scale_arr == 0):
        raise ValueError("scale must be finite and non-zero")
    if not np.all(np.isfinite(offset_arr)):
        raise ValueError("offset must be finite")
    return scale_arr, offset_arr


def _write_scaling(g: h5py.Group, scale: np.ndarray | None, offset: np.ndarray | None) -> None:
    if scale is not None:
        write_array(g, 'scale', scale.reshape(1, -1))    # MATLAB [1 x C]
        write_array(g, 'offset', offset.reshape(1, -1))  # MATLAB [1 x C]


def _read_scaling(g: h5py.Group) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The stream's `scale` and `offset` as (C,) float64, or (None, None).

    A floating `Samples` dataset never carries them: a stray pair is ignored
    with a warning.
    """
    if 'scale' not in g:
        return None, None
    C = g['Samples'].shape[0]
    if not np.issubdtype(g['Samples'].dtype, np.integer):
        warnings.warn(f"{g.file.filename}:{g.name}: scale on {g['Samples'].dtype} "
                      "Samples ignored (scaling applies to integer counts)", stacklevel=2)
        return None, None
    scale = np.asarray(g['scale'][()], dtype=np.float64).ravel()
    offset = (np.asarray(g['offset'][()], dtype=np.float64).ravel() if 'offset' in g
              else np.zeros_like(scale))
    if scale.shape != (C,) or offset.shape != (C,):
        raise ValueError(f"{g.file.filename}:{g.name}: scale/offset have "
                         f"{scale.size}/{offset.size} values for {C} channels")
    return scale, offset


def to_storage(x: np.ndarray, dtype: np.dtype, scale: np.ndarray | None = None,
               offset: np.ndarray | None = None,
               names: Sequence[str] | None = None) -> np.ndarray:
    """Cast (n, C) samples to the storage class.

    Floating input for an integer class is converted to counts,
    `rint((x - offset) / scale)` (scale 1, offset 0 when unscaled), and must
    fit the class; integer input is stored as counts as given (checked for
    range when the class differs). Floating classes just cast.
    """
    x = np.asarray(x)
    if x.dtype == dtype:
        return x
    if not np.issubdtype(dtype, np.integer):
        return x.astype(dtype)
    info = np.iinfo(dtype)
    if np.issubdtype(x.dtype, np.integer):
        counts = x.astype(np.int64)
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64)
        if scale is not None:
            y = (y - offset) / scale
        counts = np.rint(y)
    else:
        raise ValueError(f"samples must be numeric, not {x.dtype}")
    bad = ~((counts >= info.min) & (counts <= info.max))  # also True for NaN
    if np.any(bad):
        j = int(np.flatnonzero(bad.any(axis=0))[0]) if bad.ndim == 2 else 0
        ch = names[j] if names is not None and j < len(names) else f'column {j}'
        raise ValueError(f"Samples for channel {ch} do not fit {STORAGE_CLASSES[dtype]} "
                         f"counts (range {info.min} to {info.max}); use a wider dtype "
                         "or a larger scale")
    return counts.astype(dtype)


def _requested_dtype(dtype):
    """Normalise a reader's `dtype` request: None, a float dtype, or 'int'/an int dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, str) and dtype.lower() in ('int', 'counts'):
        return 'int'
    return storage_dtype(dtype)


def apply_scale(data: np.ndarray, scale: np.ndarray | None, offset: np.ndarray | None,
                dtype=None) -> np.ndarray:
    """Return the (n, C) read window in the requested form.

    `scale`/`offset` are the (C,) factors of the selected channels (None when
    the stream is unscaled). `dtype=None` gives physical float64 when scaled,
    else the data as stored; a float dtype gives physical values in that
    type; 'int' (or an integer dtype) gives the stored counts and refuses
    floating data.
    """
    req = _requested_dtype(dtype)
    if req == 'int' or (req is not None and np.issubdtype(req, np.integer)):
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError(f"the stream holds {data.dtype} samples, not integer counts")
        return data if req == 'int' or req == data.dtype else to_storage(data, req)
    if scale is None:
        return data if req is None else data.astype(req)
    phys = data.astype(np.float64) * scale + offset
    return phys if req is None or req == np.float64 else phys.astype(req)

# Regularity: within a continuous segment 0.9/SR <= dt <= 1.1/SR; a larger
# step is a gap (discontinuity), a smaller one is an error.
GAP_FACTOR = 1.1
MIN_DT_FACTOR = 0.9


# =====================================================================
# Shared pieces: dependents, sample reads
# =====================================================================


def dependents(t: np.ndarray, rate: float | None, is_regular: bool
               ) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, bool]:
    """The dependent variables of a stream from its full t vector.

    Returns `(t_coarse, df, disc, cont, is_cont)`: the coarse index and its
    decimation factor, the gap and segment tables as MATLAB sees them
    (1-indexed `[n, 2]`, empty when continuous) and the continuity flag. A gap
    is a step `> GAP_FACTOR / rate`; irregular streams are continuous by
    convention.
    """
    N = len(t)
    if is_regular and rate is not None and rate > 0:
        df = max(1, round(rate))
    else:
        span = (t[-1] - t[0]) if N > 0 else 0.0
        df = max(1, round(N / span)) if span > 0 else 1
    t_coarse = t[::df] if N > 0 else np.array([])

    disc = np.zeros((0, 2))
    cont = np.zeros((0, 2))
    is_cont = True
    if is_regular and rate is not None and rate > 0 and N > 1:
        gap_idx = np.flatnonzero(np.diff(t) > GAP_FACTOR / rate)  # 0-indexed
        if gap_idx.size:
            is_cont = False
            # 1-indexed [last_before, first_after] for MATLAB
            disc = np.column_stack([gap_idx + 1, gap_idx + 2]).astype(np.float64)
            cont = np.column_stack([np.concatenate([[1], gap_idx + 2]),
                                    np.concatenate([gap_idx + 1, [N]])]).astype(np.float64)
    return t_coarse, int(df), disc, cont, is_cont


def _write_dependent_datasets(g: h5py.Group, t_coarse: np.ndarray, df: int,
                              disc: np.ndarray, cont: np.ndarray, is_cont: bool) -> None:
    """Store the dependents in a stream group (`cont`/`disc` only when there are gaps)."""
    tc = t_coarse.reshape(-1, 1) if len(t_coarse) > 0 else np.zeros((0, 1))
    write_array(g, 't_coarse', tc)  # MATLAB [M, 1]
    write_scalar(g, 'df_t_coarse', float(df))
    if is_cont:
        for var in ('disc', 'cont'):
            if var in g:
                del g[var]
    else:
        write_array(g, 'disc', disc)  # MATLAB [n, 2]
        write_array(g, 'cont', cont)
    write_scalar(g, 'isContinuous', is_cont, logical=True)


def _resolve_channels(g: h5py.Group, channels, total_C: int,
                      names: Sequence[str] | None = None) -> np.ndarray | None:
    """Column indices of `channels`: numbers through `ch_map`, names through `names`."""
    if channels is None:
        return None
    if 'ch_map' in g:
        file_ch_map = np.squeeze(g['ch_map'][()]).ravel()
    else:
        file_ch_map = np.arange(1, total_C + 1)
    indices = []
    for ch in channels:
        if isinstance(ch, str):
            if names is None or ch not in names:
                raise ValueError(f"Channel {ch!r} not found")
            indices.append(list(names).index(ch))
        else:
            found = np.where(file_ch_map == ch)[0]
            if len(found) == 0:
                raise ValueError(f"Channel {ch} not found in ch_map")
            indices.append(int(found[0]))
    return np.array(indices, dtype=int)


def _find_range_from_time(g: h5py.Group, t_range: tuple[float, float],
                          total_N: int) -> tuple[int, int]:
    """Inclusive 0-based sample range covering `t_range`, using `t_coarse` when present."""
    if total_N == 0:
        return 0, -1

    t_start_req, t_end_req = t_range
    t_first = float(g['t'][0, 0])
    t_last = float(g['t'][0, total_N - 1])

    if t_start_req > t_last or t_end_req < t_first:
        return 0, -1  # out of range

    t_start = max(t_start_req, t_first)
    t_end = min(t_end_req, t_last)

    has_tc = 't_coarse' in g and 'df_t_coarse' in g
    if not has_tc:
        # Full t load fallback
        full_t = g['t'][0, :]
        s_start = int(np.searchsorted(full_t, t_start, side='left'))
        s_end = int(np.searchsorted(full_t, t_end, side='right') - 1)
        return s_start, s_end

    # Use t_coarse for fast lookup
    tc = g['t_coarse'][()].ravel()
    df = int(np.squeeze(g['df_t_coarse'][()]))

    # Find start
    if t_start <= t_first:
        s_start = 0
    else:
        d_idx = np.searchsorted(tc, t_start, side='left')
        ws = max(0, (d_idx - 1) * df)
        we = min((d_idx + 1) * df, total_N)
        tw = g['t'][0, ws:we]
        ri = np.searchsorted(tw, t_start, side='left')
        s_start = ws + ri

    # Find end
    if t_end >= t_last:
        s_end = total_N - 1
    else:
        d_idx = np.searchsorted(tc, t_end, side='right')
        ws = max(0, (d_idx - 1) * df)
        we = min((d_idx + 1) * df, total_N)
        tw = g['t'][0, ws:we]
        ri = np.searchsorted(tw, t_end, side='right') - 1
        s_end = ws + max(0, ri)

    s_start = max(0, min(s_start, total_N - 1))
    s_end = max(0, min(s_end, total_N - 1))
    return int(s_start), int(s_end)


def _compute_disc_info(t_out: np.ndarray, rate: float | None, is_regular: bool) -> dict:
    """Gap table of a read window (0-based indices into `t_out`)."""
    NN = len(t_out)
    result = {'is_discontinuous': False,
              'cont': np.zeros((0, 2)),
              'disc': np.zeros((0, 2))}

    if NN < 2 or not is_regular or rate is None or rate <= 0:
        if NN > 0:
            result['cont'] = np.array([[0, NN - 1]])
        return result

    gi = np.flatnonzero(np.diff(t_out) > GAP_FACTOR / rate)
    if gi.size:
        result['is_discontinuous'] = True
        result['disc'] = np.column_stack([gi, gi + 1])
        starts = np.concatenate([[0], gi + 1])
        ends = np.concatenate([gi, [NN - 1]])
        result['cont'] = np.column_stack([starts, ends])
    else:
        result['cont'] = np.array([[0, NN - 1]])
    return result


def _read_samples(g: h5py.Group, rate: float | None, is_regular: bool,
                  channels=None, t_range: tuple[float, float] | None = None,
                  s_range: tuple[int, int] | None = None,
                  channel_names: Sequence[str] | None = None, dtype=None,
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read `(Data, t_out, disc_info)` from a stream group (`t` and `Samples`).

    `channels` are numbers (through `ch_map`) or, when `channel_names` is
    given, names; `t_range` and `s_range` (0-based inclusive) are exclusive.
    `dtype` selects the form of Data (see `apply_scale`): physical float64
    when the stream has `scale`, as stored otherwise.
    """
    if t_range is not None and s_range is not None:
        raise ValueError("Specify either t_range or s_range, not both.")
    req = _requested_dtype(dtype)  # validate before touching the file
    total_N = g['t'].shape[1]
    total_C = g['Samples'].shape[0]

    channel_indices = _resolve_channels(g, channels, total_C, channel_names)
    scale, offset = _read_scaling(g)
    if channel_indices is not None and scale is not None:
        scale, offset = scale[channel_indices], offset[channel_indices]
    out_dtype = (np.float64 if scale is not None else g['Samples'].dtype) if req is None else (
        g['Samples'].dtype if req == 'int' else req)

    if s_range is not None:
        s_start, s_end = int(s_range[0]), int(s_range[1])
        if s_start < 0 or s_end >= total_N or s_start > s_end:
            raise ValueError(
                f"Invalid s_range [{s_start}, {s_end}]. Valid: [0, {total_N - 1}]")
    elif t_range is not None:
        s_start, s_end = _find_range_from_time(g, t_range, total_N)
    else:
        s_start, s_end = 0, total_N - 1

    n_ch = len(channel_indices) if channel_indices is not None else total_C
    if total_N == 0 or s_start > s_end or s_start < 0:
        empty_disc = {'is_discontinuous': False,
                      'cont': np.zeros((0, 2)), 'disc': np.zeros((0, 2))}
        return np.zeros((0, n_ch), dtype=out_dtype), np.zeros(0), empty_disc

    t_out = g['t'][0, s_start:s_end + 1]

    # Samples is HDF5 (C, N); transpose to (N, C)
    if channel_indices is not None:
        # h5py needs increasing, unique indices; reorder afterwards
        uniq, inverse = np.unique(channel_indices, return_inverse=True)
        data = g['Samples'][uniq.tolist(), s_start:s_end + 1].T[:, inverse]
    else:
        data = g['Samples'][:, s_start:s_end + 1].T

    data = apply_scale(data, scale, offset, req)
    return data, t_out, _compute_disc_info(t_out, rate, is_regular)


def _stream_info(g: h5py.Group, N: int) -> dict:
    """`isContinuous`, `t_first`, `t_last` of a stream group, when present."""
    result = {}
    if 'isContinuous' in g:
        result['isContinuous'] = bool(np.squeeze(g['isContinuous'][()]))
    if N > 0:
        result['t_first'] = float(g['t'][0, 0])
        result['t_last'] = float(g['t'][0, N - 1])
    return result


# =====================================================================
# Schema version 1: one sample rate per file
# =====================================================================


class TEA:
    """
    Time-Encoded Array — read/write timeseries HDF5/.mat files.

    A TEA object is bound to one file and holds persistent properties
    (SR, is_regular) so write/read calls don't repeat them.

    Example:
        tea = TEA('data.mat', 1000, True, t_units='s')
        tea.write(t, samples)       # creates file
        tea.write(t2, samples2)     # appends
        data, t_out, disc = tea.read(channels=[1, 3], t_range=(1.0, 2.5))
    """

    def __init__(self, file_path: str, SR: float | None, is_regular: bool, *,
                 t_units: str = '', t_offset: int | float | None = None,
                 t_offset_units: str = '', t_offset_scale: float = 1.0,
                 hdr: dict | None = None, tea_version: str = '1.0',
                 compress: bool = False,
                 expected_length: int | None = None,
                 expected_channels: int | None = None,
                 dtype=None, scale=None, offset=None) -> None:
        """
        Create or bind to a TEA file.

        Args:
            file_path: Path to the .mat file
            SR: Sample rate in samples per t_units (None for irregular data)
            is_regular: True if samples are regularly spaced
            t_units: Units of t ('s', 'us', 'ms', etc.)
            t_offset: Reference time anchor (int or float scalar)
            t_offset_units: Units of t_offset ('posix_s', 'posix_us', etc.)
            t_offset_scale: Conversion factor from t_offset units to t units
            hdr: Free-form metadata dict
            tea_version: Format version string
            compress: If True, use DEFLATE level 1 (no shuffle) on t and Samples
            expected_length: Pre-allocate N samples (avoids resize on append)
            expected_channels: Pre-allocate C channels (avoids resize on write_channels)
            dtype: Storage class of Samples: float64 (default for a new file),
                float32, int16, int32 or int64. For an existing file the stored
                class is used; giving a different one is an error.
            scale, offset: Per-channel (or one) factors for an integer dtype:
                physical = counts * scale + offset. Written on file creation
                and checked against the channel count then; readers return
                physical values by default when they are present.
        """
        self._file_path = str(file_path)
        self._SR = float(SR) if SR is not None else None
        self._is_regular = bool(is_regular)
        self.t_units = str(t_units)
        self._t_offset = t_offset
        self._t_offset_units = str(t_offset_units)
        self._t_offset_scale = float(t_offset_scale)
        self.hdr = hdr if hdr is not None else {}
        self.tea_version = str(tea_version)
        self._compress = bool(compress)
        self._expected_length = int(expected_length) if expected_length is not None else None
        self._expected_channels = int(expected_channels) if expected_channels is not None else None
        self._is_initialized = False
        self._schema_version = 1
        self._dtype = storage_dtype(dtype) if dtype is not None else np.dtype(np.float64)
        self._scale_in, self._offset_in = scale, offset  # broadcast on creation
        self._scale: np.ndarray | None = None
        self._offset: np.ndarray | None = None
        if (scale is not None or offset is not None) and not np.issubdtype(self._dtype, np.integer):
            raise ValueError(f"scale and offset need an integer dtype, not {self._dtype}")

        # Pre-allocation tracking (logical vs physical sizes)
        self._N_written: int | None = None   # logical sample count
        self._C_written: int | None = None   # logical channel count
        self._N_allocated: int | None = None  # physical allocation (N dimension)
        self._C_allocated: int | None = None  # physical allocation (C dimension)

        # Validate SR
        if is_regular:
            if SR is None or SR <= 0:
                raise ValueError("SR must be a positive number when is_regular=True")
        else:
            if SR is not None and SR <= 0:
                raise ValueError("SR must be positive or None")

        # Warn if t_units != 's' on new file
        if not os.path.isfile(self._file_path) and t_units and t_units != 's':
            warnings.warn(
                f"t_units='{t_units}'. SR is interpreted as samples/{t_units}, not Hz.",
                stacklevel=2)

        # If file exists, validate and sync
        if os.path.isfile(self._file_path):
            with h5py.File(self._file_path, 'r') as f:
                if 'SR' in f:
                    file_SR = float(np.squeeze(f['SR'][()]))
                    if SR is not None and abs(file_SR - SR) > 1e-10:
                        raise ValueError(
                            f"SR mismatch: file has {file_SR} but constructor says {SR}")
                if 'isRegular' in f:
                    file_ir = bool(np.squeeze(f['isRegular'][()]))
                    if file_ir != is_regular:
                        raise ValueError(
                            f"isRegular mismatch: file has {file_ir} but constructor says {is_regular}")
                self._is_initialized = ('t' in f and 'Samples' in f)
                self._schema_version = schema_version_of(f)
                if self._is_initialized:
                    file_dtype = f['Samples'].dtype
                    if dtype is not None and file_dtype != self._dtype:
                        raise ValueError(
                            f"dtype mismatch: file has {file_dtype} but constructor says {self._dtype}")
                    self._dtype = file_dtype
                    self._scale, self._offset = _read_scaling(f)
                    if scale is not None or offset is not None:
                        s, o = _scaling(self._dtype, scale, offset, f['Samples'].shape[0])
                        if self._scale is None or not (np.array_equal(s, self._scale)
                                                       and np.array_equal(o, self._offset)):
                            raise ValueError("scale/offset mismatch with the existing file")

    # ============ Properties ============

    @property
    def dtype(self) -> np.dtype:
        """Storage class of Samples (the file's once it exists)."""
        return self._dtype

    @property
    def scale(self) -> np.ndarray | None:
        """Per-channel scale of an integer file, (C,) float64, or None."""
        return None if self._scale is None else self._scale.copy()

    @property
    def offset(self) -> np.ndarray | None:
        """Per-channel offset of an integer file, (C,) float64, or None."""
        return None if self._offset is None else self._offset.copy()

    @property
    def schema_version(self) -> int:
        """Schema version of the bound file (1 for a new or version-1 file)."""
        return self._schema_version

    @property
    def rates(self) -> list[float]:
        """Sample rates of the streams in the file (one for a version-1 file)."""
        if not os.path.isfile(self._file_path):
            return []
        return read(self._file_path).rates

    def stream(self, rate: float | None = None) -> 'TEAStream':
        """Read-only view of the stream with this sample rate.

        `rate` may be omitted when the file holds one stream.
        """
        return read(self._file_path).stream(rate)

    @property
    def file_path(self) -> str:
        """Bound file path (immutable)."""
        return self._file_path

    @property
    def SR(self) -> float | None:
        """Sample rate in samples per t_units (immutable)."""
        return self._SR

    @property
    def is_regular(self) -> bool:
        """Regularity flag (immutable)."""
        return self._is_regular

    @property
    def N(self) -> int:
        """Total sample count (logical, not physical allocation)."""
        if self._N_written is not None:
            return self._N_written
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return 0
        with h5py.File(self._file_path, 'r') as f:
            return f['t'].shape[1]

    @property
    def C(self) -> int:
        """Channel count (logical, not physical allocation)."""
        if self._C_written is not None:
            return self._C_written
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return 0
        with h5py.File(self._file_path, 'r') as f:
            return f['Samples'].shape[0]

    @property
    def ch_map(self) -> ArrayFloat64:
        """Channel identity vector (read from file)."""
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return np.array([], dtype=np.float64)
        with h5py.File(self._file_path, 'r') as f:
            if 'ch_map' in f:
                return np.squeeze(f['ch_map'][()]).ravel().astype(np.float64)
            return np.arange(1, self.C + 1, dtype=np.float64)

    def _refuse_v2_write(self) -> None:
        if self._schema_version >= 2:
            raise RuntimeError(
                f"{self._file_path} is a TEA version-{self._schema_version} file; "
                "write it with tea.write() or tea.Writer, not TEA.write().")

    # ============ Write ============

    def write(self, t: npt.ArrayLike, samples: npt.ArrayLike) -> None:
        """
        Append time-series data to the TEA file.

        On first write, creates the file and writes metadata.
        On subsequent writes, appends in time.

        Args:
            t: 1D array of timestamps (monotonically increasing)
            samples: 2D array of shape (N, C)
        """
        self._refuse_v2_write()
        t = np.asarray(t, dtype=np.float64).ravel()
        samples = np.atleast_2d(np.asarray(samples))
        if samples.ndim == 2 and samples.shape[0] == 1 and len(t) > 1:
            samples = samples.T

        N_new = len(t)
        self._validate_t(t)
        if samples.shape[0] != N_new:
            raise ValueError(
                f"Samples must have {N_new} rows to match t, but has {samples.shape[0]}")
        C_new = samples.shape[1]
        if not self._is_initialized and self._scale is None:
            self._scale, self._offset = _scaling(self._dtype, self._scale_in, self._offset_in, C_new)
        if self._scale is not None and C_new != len(self._scale):
            raise ValueError(
                f"Channel count mismatch: file has {len(self._scale)} but write has {C_new}.")
        samples = to_storage(samples, self._dtype, self._scale, self._offset)

        if self._is_regular:
            self._validate_regularity(t)

        # --- Sanity checks ---
        # Large-t-with-offset: warn if t looks absolute
        if self._t_offset is not None and abs(t[0]) > 1e8:
            warnings.warn(
                f"t[0]={t[0]:.3e} looks like an absolute timestamp. "
                f"Use write_absolute() or subtract t_offset.",
                stacklevel=2)
        # Float64 precision check
        if N_new > 1:
            ulp_val = np.spacing(abs(t[0]))
            median_dt = np.median(np.diff(t))
            if median_dt > 0 and (ulp_val / median_dt) > 0.01:
                warnings.warn(
                    f"float64 precision at t[0]={t[0]:.3e} is {ulp_val:.3e} ULP, "
                    f"which is {ulp_val / median_dt * 100:.1f}% of sample spacing. "
                    f"Consider using t_offset.",
                    stacklevel=2)

        if not self._is_initialized:
            self._create_file(t, samples)
            self._is_initialized = True
            logger.info("TEA file created: %s (%d samples, %d channels)",
                        self._file_path, N_new, C_new)
        else:
            self._append_time(t, samples)

    # ============ Write Absolute ============

    def write_absolute(self, t_abs: npt.ArrayLike, samples: npt.ArrayLike) -> None:
        """
        Write using absolute timestamps — subtracts t_offset * t_offset_scale.

        Args:
            t_abs: 1D array of absolute timestamps
            samples: 2D array of shape (N, C)
        """
        if self._t_offset is None:
            raise ValueError(
                "Cannot use write_absolute() without t_offset. "
                "Set t_offset in constructor.")
        t_rel = np.float64(t_abs) - np.float64(self._t_offset) * self._t_offset_scale
        self.write(t_rel, samples)

    # ============ Write Channels ============

    def write_channels(self, samples: npt.ArrayLike,
                       ch_map: list[int] | npt.ArrayLike | None = None) -> None:
        """
        Append new channels (columns) to existing data.

        Args:
            samples: 2D array (N, C_new)
            ch_map: Channel identity for new channels (optional)
        """
        if not self._is_initialized:
            raise RuntimeError("Cannot append channels before any data has been written.")
        self._refuse_v2_write()
        if self._scale is not None:
            raise ValueError("write_channels is not supported on a file with per-channel "
                             "scale/offset; write all channels in write()")

        samples = to_storage(np.atleast_2d(np.asarray(samples)), self._dtype)

        with h5py.File(self._file_path, 'r+') as f:
            N_written = self._N_written if self._N_written is not None else f['t'].shape[1]
            C_old = self._C_written if self._C_written is not None else f['Samples'].shape[0]
            N_new, C_new = samples.shape

            if N_new != N_written:
                raise ValueError(f"New channels must have {N_written} rows, but has {N_new}")

            C_total = C_old + C_new
            C_phys = self._C_allocated or f['Samples'].shape[0]
            N_phys = self._N_allocated or f['Samples'].shape[1]

            # Resize only if we exceed the allocated channel space
            if C_total > C_phys:
                f['Samples'].resize((C_total, N_phys))
                self._C_allocated = C_total

            f['Samples'][C_old:C_total, :N_written] = samples.T

            # Update ch_map
            if 'ch_map' in f:
                old_map = np.squeeze(f['ch_map'][()]).ravel()
            else:
                old_map = np.arange(1, C_old + 1, dtype=np.float64)

            if ch_map is not None:
                new_map = np.concatenate([old_map, np.asarray(ch_map, dtype=np.float64).ravel()])
            else:
                new_map = np.concatenate([old_map, np.arange(C_old + 1, C_total + 1, dtype=np.float64)])

            write_array(f, 'ch_map', new_map.reshape(1, -1))  # MATLAB [1, C]

        self._C_written = C_total
        logger.info("TEA channels appended: %s (now %d channels)",
                    self._file_path, C_total)

    # ============ Finalize ============

    def finalize(self) -> None:
        """
        Trim pre-allocated datasets to actual written size and update dependents.

        Must be called after all writes are complete when using expected_length
        or expected_channels. Safe to call even without pre-allocation (no-op).
        """
        if not self._is_initialized:
            return

        N = self._N_written
        C = self._C_written
        if N is None or C is None:
            return  # nothing to finalize

        with h5py.File(self._file_path, 'r+') as f:
            N_phys = f['t'].shape[1]
            C_phys = f['Samples'].shape[0]

            needs_trim = (N_phys != N) or (C_phys != C)
            if needs_trim:
                f['t'].resize((1, N))
                f['Samples'].resize((C, N))
                logger.info("TEA finalized: trimmed from (%d, %d) to (%d, %d)",
                            C_phys, N_phys, C, N)

            # Rewrite dependents based on actual data
            t = f['t'][0, :N]
            self._write_dependents(f, t)

        self._N_allocated = N
        self._C_allocated = C

        logger.info("TEA finalized: %s (%d samples, %d channels)",
                    self._file_path, N, C)

    # ============ Read ============

    def read(self, channels: list[int] | None = None,
             t_range: tuple[float, float] | None = None,
             s_range: tuple[int, int] | None = None, *, dtype=None,
             ) -> tuple[np.ndarray, ArrayFloat64, dict]:
        """
        Read data from the TEA file.

        Args:
            channels: Channel numbers (uses ch_map). None for all.
            t_range: (start, end) in t units. None to skip.
            s_range: (start, end) sample indices, 0-indexed inclusive. None to skip.
            dtype: None (default): physical float64 when the file has `scale`,
                else as stored; np.float32 / np.float64: physical values in that
                type; 'int' (or an integer dtype): the stored counts, an error
                for a floating file.

        Returns:
            (Data, t_out, disc_info) tuple.
            Data: (N, C) array. t_out: 1D array. disc_info: dict.
        """
        if t_range is not None and s_range is not None:
            raise ValueError("Specify either t_range or s_range, not both.")

        with h5py.File(self._file_path, 'r') as f:
            if 't' not in f or 'Samples' not in f:
                raise ValueError(
                    f"{self._file_path} holds several sample rates "
                    f"({', '.join(f'{r:g}' for r in read(self._file_path).rates)}); "
                    "read one with .stream(rate).read(...)")
            return _read_samples(f, self._SR, self._is_regular, channels, t_range, s_range,
                                 dtype=dtype)

    # ============ Refresh ============

    def refresh(self) -> None:
        """Recompute all dependent variables from the full t vector."""
        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(f"File not found: {self._file_path}")
        self._refuse_v2_write()

        with h5py.File(self._file_path, 'r+') as f:
            t = f['t'][0, :]
            self._write_dependents(f, t)
            if 'tea_version' not in f:
                write_string(f, 'tea_version', '1.0')

        logger.info("TEA refreshed: %s", self._file_path)

    # ============ Info ============

    def info(self) -> dict:
        """Return a summary dict of the TEA file."""
        result = {
            'file_path': self._file_path,
            'SR': self._SR,
            'is_regular': self._is_regular,
            'N': self.N,
            'C': self.C,
            'ch_map': self.ch_map,
            't_units': self.t_units,
            't_offset': self._t_offset,
            't_offset_units': self._t_offset_units,
            't_offset_scale': self._t_offset_scale,
            'schema_version': self._schema_version,
            'dtype': STORAGE_CLASSES.get(self._dtype, str(self._dtype)),
            'scaled': self._scale is not None,
        }
        if os.path.isfile(self._file_path) and self._is_initialized:
            with h5py.File(self._file_path, 'r') as f:
                result.update(_stream_info(f, result['N']))
        return result

    # ============ Private: Validation ============

    def _validate_t(self, t: ArrayFloat64) -> None:
        if len(t) == 0:
            raise ValueError("t must not be empty")
        if not np.issubdtype(t.dtype, np.number):
            raise ValueError("t must be numeric")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly monotonically increasing")

    def _validate_regularity(self, t: ArrayFloat64) -> None:
        """Within a segment 0.9/SR <= dt <= 1.1/SR; at least half the steps are within segments."""
        if len(t) < 2:
            return
        dt = np.diff(t)
        expected_dt = 1.0 / self._SR
        if np.any(dt < MIN_DT_FACTOR * expected_dt):
            raise ValueError(
                f"is_regular=True but samples are closer than {MIN_DT_FACTOR} / SR "
                f"(min dt {dt.min():.6g}, 1/SR {expected_dt:.6g})")
        regular_mask = dt <= GAP_FACTOR * expected_dt
        regular_fraction = np.sum(regular_mask) / len(dt)
        if regular_fraction < 0.5:
            raise ValueError(
                f"is_regular=True but only {regular_fraction * 100:.0f}% of dt values are within 1.1/SR")

    # ============ Private: File Creation ============

    def _create_file(self, t: ArrayFloat64, samples: ArrayFloat64) -> None:
        N, C = len(t), samples.shape[1]

        # Determine allocation sizes
        N_alloc = max(N, self._expected_length or N)
        C_alloc = max(C, self._expected_channels or C)
        chunk_c = max(1, C_alloc)
        chunk_n = min(chunk_samples(chunk_c, self._dtype), max(1, N_alloc))
        chunk_t = min(chunk_samples(1), max(1, N_alloc))

        # Filters: DEFLATE 1 when compressed; shuffle for integer classes only
        comp_kwargs = compression_kwargs(self._compress)
        comp_samples = compression_kwargs(self._compress, self._dtype)

        # Create HDF5 file with 512-byte userblock for MATLAB v7.3 compatibility
        with h5py.File(self._file_path, 'w', userblock_size=512) as f:
            # SR: scalar for regular, empty [] for irregular
            if self._SR is not None:
                write_scalar(f, 'SR', np.float64(self._SR))
            else:
                ds_sr = f.create_dataset('SR', data=np.array([0, 0], dtype=np.uint64))
                ds_sr.attrs['MATLAB_class'] = np.bytes_('double')
                ds_sr.attrs['MATLAB_empty'] = np.uint8(1)
            # isRegular
            write_scalar(f, 'isRegular', self._is_regular, logical=True)
            # t: HDF5 (1, N_alloc), write first N values
            ds_t = f.create_dataset('t', shape=(1, N_alloc),
                                    maxshape=(1, None), chunks=(1, chunk_t),
                                    dtype=np.float64, **comp_kwargs)
            ds_t.attrs['MATLAB_class'] = np.bytes_('double')
            ds_t[0, :N] = t
            # Samples: HDF5 (C_alloc, N_alloc), write first (C, N) values
            ds_s = f.create_dataset('Samples', shape=(C_alloc, N_alloc),
                                    maxshape=(None, None), chunks=(chunk_c, chunk_n),
                                    dtype=self._dtype, **comp_samples)
            ds_s.attrs['MATLAB_class'] = np.bytes_(STORAGE_CLASSES[self._dtype])
            ds_s[:C, :N] = samples.T
            # String metadata
            write_string(f, 'tea_version', self.tea_version)
            if self.t_units:
                write_string(f, 't_units', self.t_units)
            # t_offset metadata
            if self._t_offset is not None:
                write_scalar(f, 't_offset', self._t_offset)
                write_string(f, 't_offset_units', self._t_offset_units)
                write_scalar(f, 't_offset_scale', float(self._t_offset_scale))
            # Per-channel scale and offset of integer counts
            _write_scaling(f, self._scale, self._offset)
            # Dependent variables
            self._write_dependents(f, t)

            # hdr: free-form metadata stored as MATLAB struct group
            if self.hdr:
                write_hdr_fields(f, self.hdr)

        # Write the 128-byte MATLAB v7.3 header into the userblock
        write_matlab_header(self._file_path)

        # Track logical and physical sizes
        self._N_written = N
        self._C_written = C
        self._N_allocated = N_alloc
        self._C_allocated = C_alloc

    # ============ Private: Append ============

    def _append_time(self, t: ArrayFloat64, samples: ArrayFloat64) -> None:
        N_new, C_new = len(t), samples.shape[1]

        with h5py.File(self._file_path, 'r+') as f:
            # Use logical sizes (handles pre-allocated files)
            N_old = self._N_written if self._N_written is not None else f['t'].shape[1]
            C_old = self._C_written if self._C_written is not None else f['Samples'].shape[0]

            if C_new != C_old:
                raise ValueError(
                    f"Channel count mismatch: file has {C_old} but write has {C_new}. "
                    "Use write_channels to add columns.")

            t_last = float(f['t'][0, N_old - 1])
            if t[0] <= t_last:
                raise ValueError(
                    f"New t must start after existing t. Last: {t_last}, first new: {t[0]}")

            N_total = N_old + N_new
            N_phys = self._N_allocated or f['t'].shape[1]

            # Resize only if we exceed the allocated space
            if N_total > N_phys:
                f['t'].resize((1, N_total))
                f['Samples'].resize((f['Samples'].shape[0], N_total))
                self._N_allocated = N_total

            f['t'][0, N_old:N_total] = t
            f['Samples'][:C_old, N_old:N_total] = samples.T

            # Update dependents incrementally
            self._update_dependents_incremental(f, t, t_last, N_old, N_total)

        self._N_written = N_total
        logger.info("TEA file appended: %s (now %d samples)",
                    self._file_path, N_total)

    # ============ Private: Dependents ============

    def _write_dependents(self, f: h5py.File, t: ArrayFloat64) -> None:
        """Recompute and store every dependent from the full t vector."""
        _write_dependent_datasets(f, *dependents(t, self._SR, self._is_regular))

    def _update_dependents_incremental(self, f: h5py.File, t_new: ArrayFloat64,
                                       t_last: float, N_old: int,
                                       N_total: int) -> None:
        N_new = len(t_new)

        # Check if we have prior metadata
        if 'isContinuous' not in f or 'df_t_coarse' not in f:
            # Fallback to full recompute
            t_full = f['t'][0, :]
            self._write_dependents(f, t_full)
            return

        # === isContinuous / cont / disc ===
        if self._is_regular and self._SR is not None and self._SR > 0:
            gap_threshold = GAP_FACTOR / self._SR

            old_is_cont = bool(np.squeeze(f['isContinuous'][()]))
            if not old_is_cont and 'disc' in f:
                old_disc = f['disc'][()].T  # HDF5 transposed → MATLAB shape (n,2)
            else:
                old_disc = np.zeros((0, 2))

            # Check junction gap
            new_gaps = []
            if (t_new[0] - t_last) > gap_threshold:
                new_gaps.append([N_old, N_old + 1])  # 1-indexed

            # Check gaps within new chunk
            if N_new > 1:
                dt_new = np.diff(t_new)
                gap_mask = dt_new > gap_threshold
                if np.any(gap_mask):
                    gi = np.where(gap_mask)[0]
                    for g in gi:
                        new_gaps.append([N_old + g + 1, N_old + g + 2])  # 1-indexed

            if len(new_gaps) > 0:
                new_gap_arr = np.array(new_gaps, dtype=np.float64)
            else:
                new_gap_arr = np.zeros((0, 2))

            all_disc = np.vstack([old_disc, new_gap_arr]) if (old_disc.size > 0 or new_gap_arr.size > 0) else np.zeros((0, 2))

            if all_disc.size == 0 or all_disc.shape[0] == 0:
                write_scalar(f, 'isContinuous', True, logical=True)
                for var in ['disc', 'cont']:
                    if var in f:
                        del f[var]
            else:
                write_scalar(f, 'isContinuous', False, logical=True)
                write_array(f, 'disc', all_disc)  # MATLAB [n, 2]
                starts = np.concatenate([[1], all_disc[:, 1]])
                ends = np.concatenate([all_disc[:, 0], [N_total]])
                cont = np.column_stack([starts, ends])
                write_array(f, 'cont', cont)  # MATLAB [n, 2]
        else:
            write_scalar(f, 'isContinuous', True, logical=True)

        # === t_coarse ===
        df = int(np.squeeze(f['df_t_coarse'][()]))
        if 't_coarse' in f:
            # t_coarse stored as HDF5 transposed; read and flatten
            old_tc = f['t_coarse'][()].ravel()
        else:
            old_tc = np.array([])

        last_coarse_global = 1 + ((N_old - 1) // df) * df  # 1-indexed
        next_coarse_global = last_coarse_global + df
        new_coarse_globals = np.arange(next_coarse_global, N_total + 1, df)

        if len(new_coarse_globals) > 0:
            new_coarse_locals = (new_coarse_globals - N_old - 1).astype(int)  # 0-indexed into t_new
            valid = new_coarse_locals < N_new
            new_coarse_locals = new_coarse_locals[valid]
            new_tc = t_new[new_coarse_locals]
            combined_tc = np.concatenate([old_tc, new_tc])
            write_array(f, 't_coarse', combined_tc.reshape(-1, 1))  # MATLAB [M, 1]

    # ============ HDF5 helpers (kept as aliases of tea_h5) ============

    _write_h5_scalar = staticmethod(write_scalar)
    _write_h5_array = staticmethod(write_array)
    _write_h5_string = staticmethod(write_string)
    _write_hdr_fields = staticmethod(write_hdr_fields)
    _write_matlab_header = staticmethod(write_matlab_header)


# =====================================================================
# Schema version 2: several sample rates in one file
# =====================================================================


def _find_rate(rates: Sequence[float | None], rate: float | None) -> int:
    """Index of `rate` in `rates`; `rate=None` picks the only stream."""
    if rate is None:
        if len(rates) == 1:
            return 0
        raise ValueError(f"The file holds {len(rates)} streams; give a rate "
                         f"({', '.join(f'{r:g}' for r in rates if r is not None)}).")
    for i, r in enumerate(rates):
        if r is not None and abs(r - rate) <= 1e-9 * max(1.0, abs(rate)):
            return i
    raise KeyError(f"No stream with rate {rate:g}; the file has "
                   f"{', '.join(f'{r:g}' for r in rates if r is not None) or 'none'}.")


@dataclass
class Stream:
    """One sample rate's data, as given to `write`.

    Attributes:
        rate: Sample rate in samples per t unit (Hz when t is in seconds).
        channel_names: One name per column of `samples`.
        units: One unit string per channel, or one string for all (e.g. 'uV'):
            the physical unit, after scaling when `scale` is given.
        samples: (n, ch) array; converted to `dtype` (see `Writer.append`).
        t: (n,) timestamps, strictly increasing, relative to t_offset. Gaps are
            allowed: a step larger than 1.1 / rate starts a new segment.
        is_regular: True when samples are evenly spaced within segments.
        dtype: Storage class: float64 (default), float32, int16, int32, int64.
        scale, offset: Per-channel (or one) factors for an integer dtype,
            physical = counts * scale + offset.
    """
    rate: float
    channel_names: Sequence[str]
    units: str | Sequence[str]
    samples: npt.ArrayLike
    t: npt.ArrayLike
    is_regular: bool = True
    dtype: npt.DTypeLike = np.float64
    scale: float | Sequence[float] | None = None
    offset: float | Sequence[float] | None = None


class TEAStream:
    """Read-only view of one stream (one sample rate) of a TEA file.

    Has the `TEA` read API (`read`, `N`, `C`, `ch_map`, `info`) plus `rate`,
    `channel_names`, `units`, `dtype` (stored class), `scale` and `offset`.
    `read(channels=...)` accepts channel numbers (1-based, or the file's
    ch_map) and channel names; `read(dtype=...)` picks physical values or
    counts (see `apply_scale`).
    """

    def __init__(self, file_path: str, group: str, rate: float | None,
                 is_regular: bool, channel_names: list[str], units: list[str], *,
                 schema_version: int, t_units: str = '',
                 t_offset: int | float | None = None, t_offset_units: str = '',
                 t_offset_scale: float = 1.0, dtype=np.float64,
                 scale: np.ndarray | None = None, offset: np.ndarray | None = None) -> None:
        self._file_path = str(file_path)
        self._group = group
        self._schema_version = int(schema_version)
        self._SR = float(rate) if rate is not None else None
        self._is_regular = bool(is_regular)
        self.t_units = t_units
        self._t_offset = t_offset
        self._t_offset_units = t_offset_units
        self._t_offset_scale = float(t_offset_scale)
        self._channel_names = list(channel_names)
        self._units = list(units)
        self._dtype = np.dtype(dtype)
        self._scale = scale
        self._offset = offset

    @property
    def dtype(self) -> np.dtype:
        """Storage class of Samples (what `read(dtype='int')` returns for counts)."""
        return self._dtype

    @property
    def scale(self) -> np.ndarray | None:
        """Per-channel scale (C,) of an integer stream, or None when unscaled."""
        return None if self._scale is None else self._scale.copy()

    @property
    def offset(self) -> np.ndarray | None:
        """Per-channel offset (C,) of an integer stream, or None when unscaled."""
        return None if self._offset is None else self._offset.copy()

    def __repr__(self) -> str:
        return (f"TEAStream({self._file_path!r}, group={self._group!r}, "
                f"rate={self._SR}, N={self.N}, C={self.C})")

    # ---- properties ----

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def group(self) -> str:
        """HDF5 group holding this stream's t and Samples."""
        return self._group

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def rate(self) -> float | None:
        """Sample rate of this stream."""
        return self._SR

    SR = rate

    @property
    def is_regular(self) -> bool:
        return self._is_regular

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def units(self) -> list[str]:
        return list(self._units)

    @property
    def t_offset(self):
        return self._t_offset

    @property
    def t_offset_units(self) -> str:
        return self._t_offset_units

    @property
    def t_offset_scale(self) -> float:
        return self._t_offset_scale

    @property
    def N(self) -> int:
        """Sample count."""
        with h5py.File(self._file_path, 'r') as f:
            return f[self._group]['t'].shape[1]

    @property
    def C(self) -> int:
        """Channel count."""
        with h5py.File(self._file_path, 'r') as f:
            return f[self._group]['Samples'].shape[0]

    @property
    def ch_map(self) -> ArrayFloat64:
        """Channel identity vector (`ch_map` when stored, else 1..C)."""
        with h5py.File(self._file_path, 'r') as f:
            g = f[self._group]
            if 'ch_map' in g:
                return np.squeeze(g['ch_map'][()]).ravel().astype(np.float64)
            return np.arange(1, g['Samples'].shape[0] + 1, dtype=np.float64)

    # ---- read ----

    def read(self, channels=None, t_range: tuple[float, float] | None = None,
             s_range: tuple[int, int] | None = None, *, dtype=None,
             ) -> tuple[np.ndarray, ArrayFloat64, dict]:
        """Read `(Data, t_out, disc_info)`; `channels` by number (ch_map) or name.

        `dtype`: None gives physical float64 when the stream has `scale`, else
        the samples as stored; np.float32 / np.float64 give physical values in
        that type; 'int' (or an integer dtype) gives the stored counts and is
        an error for a floating stream.
        """
        with h5py.File(self._file_path, 'r') as f:
            return _read_samples(f[self._group], self._SR, self._is_regular,
                                 channels, t_range, s_range, self._channel_names, dtype)

    def info(self) -> dict:
        with h5py.File(self._file_path, 'r') as f:
            g = f[self._group]
            N, C = g['t'].shape[1], g['Samples'].shape[0]
            result = {
                'file_path': self._file_path,
                'SR': self._SR,
                'is_regular': self._is_regular,
                'N': N,
                'C': C,
                'ch_map': self.ch_map,
                't_units': self.t_units,
                't_offset': self._t_offset,
                't_offset_units': self._t_offset_units,
                't_offset_scale': self._t_offset_scale,
                'schema_version': self._schema_version,
                'dtype': STORAGE_CLASSES.get(self._dtype, str(self._dtype)),
                'scaled': self._scale is not None,
            }
            result.update(_stream_info(g, N))
        result.update(rate=self._SR, channel_names=self.channel_names,
                      units=self.units, group=self._group)
        return result

    def _read_only(self, *args, **kwargs):
        raise RuntimeError("A TEAStream is read-only; write with tea.write() or tea.Writer.")

    write = write_absolute = write_channels = finalize = refresh = _read_only


class TEAFile:
    """A TEA file of any schema version, opened for reading with `read(path)`.

    Attributes:
        file_path, schema_version, rates, channel_names (all streams, in
        stream order), channel_rates (rate of each of those channels),
        channel_scales (scale of each, None for unscaled streams),
        t_units, t_offset, t_offset_units, t_offset_scale.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = str(file_path)
        with h5py.File(self.file_path, 'r') as f:
            self.schema_version = schema_version_of(f)
            if self.schema_version > SCHEMA_VERSION:
                raise ValueError(
                    f"{self.file_path} is TEA schema version {self.schema_version}; "
                    f"this reader knows up to {SCHEMA_VERSION}.")
            self.t_units = read_string(f, 't_units')
            self.t_offset = read_scalar(f, 't_offset')
            self.t_offset_units = read_string(f, 't_offset_units')
            self.t_offset_scale = float(read_scalar(f, 't_offset_scale', 1.0))
            self._streams = []
            if self.schema_version >= 2:
                groups = sorted(f[RATES_GROUP].keys(), key=lambda k: int(k[1:]))
                for k in groups:
                    g = f[RATES_GROUP][k]
                    self._streams.append(self._make_stream(
                        f'/{RATES_GROUP}/{k}', g,
                        read_char_matrix(g['channel_names']),
                        read_char_matrix(g['units'])))
            elif 't' in f and 'Samples' in f:
                C = f['Samples'].shape[0]
                names = None
                if 'hdr' in f and 'channel_names' in f['hdr']:
                    names = read_char_matrix(f['hdr']['channel_names'])
                if names is None or len(names) != C:
                    if 'ch_map' in f:
                        ch_map = np.squeeze(f['ch_map'][()]).ravel()
                    else:
                        ch_map = np.arange(1, C + 1)
                    names = [f'Ch{int(n)}' for n in ch_map]
                self._streams.append(self._make_stream('/', f, names, [''] * C))

    def _make_stream(self, group: str, g: h5py.Group, names, units) -> TEAStream:
        scale, offset = _read_scaling(g)
        return TEAStream(
            self.file_path, group, read_scalar(g, 'SR'),
            bool(read_scalar(g, 'isRegular', True)), names, units,
            schema_version=self.schema_version, t_units=self.t_units,
            t_offset=self.t_offset, t_offset_units=self.t_offset_units,
            t_offset_scale=self.t_offset_scale, dtype=g['Samples'].dtype,
            scale=scale, offset=offset)

    def __repr__(self) -> str:
        return (f"TEAFile({self.file_path!r}, schema_version={self.schema_version}, "
                f"rates={self.rates})")

    @property
    def rates(self) -> list[float]:
        """Sample rate of each stream, in file order."""
        return [s.rate for s in self._streams]

    @property
    def streams(self) -> list[TEAStream]:
        return list(self._streams)

    @property
    def channel_names(self) -> list[str]:
        return [n for s in self._streams for n in s.channel_names]

    @property
    def channel_rates(self) -> list[float]:
        return [s.rate for s in self._streams for _ in s.channel_names]

    @property
    def channel_scales(self) -> list[float | None]:
        """Scale of each channel (counts to physical), None for unscaled streams."""
        return [None if s.scale is None else float(s.scale[i])
                for s in self._streams for i in range(len(s.channel_names))]

    def stream(self, rate: float | None = None) -> TEAStream:
        """The stream with this sample rate (`rate` may be omitted for one stream)."""
        return self._streams[_find_rate(self.rates, rate)]


def read(file_path: str | os.PathLike) -> TEAFile:
    """Open a TEA file of schema version 1 or 2 for reading.

    Example:
        f = tea.read('clip.mat')
        f.rates                                   # [256.0, 1024.0]
        data, t, disc = f.stream(1024).read(t_range=(10.0, 12.0))
    """
    return TEAFile(os.fspath(file_path))


class _StreamState:
    """Writer bookkeeping for one stream."""

    def __init__(self, rate: float, is_regular: bool, names: list[str],
                 units: list[str], group: h5py.Group, dtype: np.dtype,
                 scale: np.ndarray | None, offset: np.ndarray | None) -> None:
        self.rate = rate
        self.is_regular = is_regular
        self.names = names
        self.units = units
        self.g = group
        self.dtype = dtype
        self.scale = scale
        self.offset = offset
        self.n = 0
        self.t_last: float | None = None
        self.disc: list[tuple[int, int]] = []  # 1-indexed [last_before, first_after]
        self.df = max(1, round(rate)) if is_regular else 1
        self.t_coarse: list[np.ndarray] = []


class Writer:
    """Write a TEA schema-version-2 file, appending to each stream in pieces.

    Example:
        with tea.Writer('clip.mat', t_offset=1770000000, compress=True) as w:
            w.add_stream(256, ['Fp1', 'Fp2'], 'uV')
            w.add_stream(1024, ['DC1'], 'mV')
            for t, x in blocks_256:  w.append(256, t, x)
            for t, x in blocks_1024: w.append(1024, t, x)

    Each stream lives in its own group `/rates/r<k>` (k = 1, 2, ... in the
    order streams are added) with the version-1 layout: `t`, `Samples`, `SR`,
    `isRegular`, `t_coarse`, `df_t_coarse`, `isContinuous`, `cont`, `disc`,
    plus `channel_names` and `units`. The file is finished by `close()` (or
    leaving the `with` block); a file with one stream then also gets the
    version-1 fields at the top level.
    """

    def __init__(self, file_path: str | os.PathLike, *,
                 t_offset: int | float | None = None, t_offset_units: str = 'posix_s',
                 t_offset_scale: float = 1.0, t_units: str = 's',
                 hdr: dict | None = None, compress: bool = False) -> None:
        self.file_path = os.fspath(file_path)
        self._t_offset = t_offset
        self._t_offset_units = t_offset_units
        self._t_offset_scale = float(t_offset_scale)
        self._t_units = t_units
        self._hdr = hdr or {}
        self._compress = bool(compress)
        self._streams: list[_StreamState] = []
        self._f: h5py.File | None = h5py.File(self.file_path, 'w', userblock_size=512)
        rates = self._f.create_group(RATES_GROUP)
        rates.attrs['MATLAB_class'] = np.bytes_('struct')
        write_scalar(self._f, 'schema_version', float(SCHEMA_VERSION))

    def __enter__(self) -> 'Writer':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif self._f is not None:
            self._f.close()
            self._f = None

    @property
    def rates(self) -> list[float]:
        return [s.rate for s in self._streams]

    def add_stream(self, rate: float, channel_names: Sequence[str],
                   units: str | Sequence[str], *, is_regular: bool = True,
                   dtype=np.float64, scale=None, offset=None) -> None:
        """Declare a stream; its data follows through `append(rate, t, samples)`.

        `dtype` is the storage class of Samples (float64, float32, int16, int32
        or int64). `scale` and `offset` (one value or one per channel) are
        allowed with an integer dtype only: physical = counts * scale + offset,
        and `units` names the physical unit.
        """
        if self._f is None:
            raise RuntimeError("Writer is closed")
        rate = float(rate)
        dtype = storage_dtype(dtype)
        if not rate > 0:
            raise ValueError("rate must be positive")
        if any(abs(r - rate) <= 1e-9 * max(1.0, rate) for r in self.rates):
            raise ValueError(f"A stream with rate {rate:g} already exists")
        names = [str(n) for n in channel_names]
        if not names:
            raise ValueError("A stream needs at least one channel")
        unit_list = [units] * len(names) if isinstance(units, str) else [str(u) for u in units]
        if len(unit_list) != len(names):
            raise ValueError(f"{len(unit_list)} units for {len(names)} channels")

        C = len(names)
        scale_arr, offset_arr = _scaling(dtype, scale, offset, C)
        g = self._f[RATES_GROUP].create_group(f'r{len(self._streams) + 1}')
        g.attrs['MATLAB_class'] = np.bytes_('struct')
        ds_t = g.create_dataset('t', shape=(1, 0), maxshape=(1, None),
                                chunks=(1, chunk_samples(1)), dtype=np.float64,
                                **compression_kwargs(self._compress))
        ds_t.attrs['MATLAB_class'] = np.bytes_('double')
        ds_s = g.create_dataset('Samples', shape=(C, 0), maxshape=(C, None),
                                chunks=(C, chunk_samples(C, dtype)), dtype=dtype,
                                **compression_kwargs(self._compress, dtype))
        ds_s.attrs['MATLAB_class'] = np.bytes_(STORAGE_CLASSES[dtype])
        write_scalar(g, 'SR', rate)
        write_scalar(g, 'isRegular', is_regular, logical=True)
        write_char_matrix(g, 'channel_names', names)
        write_char_matrix(g, 'units', unit_list)
        _write_scaling(g, scale_arr, offset_arr)
        self._streams.append(_StreamState(rate, bool(is_regular), names, unit_list, g,
                                          dtype, scale_arr, offset_arr))

    def append(self, rate: float, t: npt.ArrayLike, samples: npt.ArrayLike) -> None:
        """Append samples (n, ch) with timestamps t (n,) to the stream of this rate.

        Samples are converted to the stream's dtype: for an integer dtype,
        floating input is `rint((x - offset) / scale)` and must fit the type
        (ValueError on overflow, naming the channel), integer input is stored
        as the counts given; a floating dtype casts.
        """
        if self._f is None:
            raise RuntimeError("Writer is closed")
        st = self._streams[_find_rate(self.rates, float(rate))]
        t = np.asarray(t, dtype=np.float64).ravel()
        samples = np.asarray(samples)
        if samples.ndim == 1 and len(st.names) == 1:
            samples = samples.reshape(-1, 1)
        n = len(t)
        if n == 0:
            return
        if samples.shape != (n, len(st.names)):
            raise ValueError(f"samples has shape {samples.shape}; expected ({n}, {len(st.names)})")
        samples = to_storage(samples, st.dtype, st.scale, st.offset, st.names)
        tt = t if st.t_last is None else np.concatenate([[st.t_last], t])
        dt = np.diff(tt)
        if np.any(dt <= 0):
            raise ValueError("t must be strictly increasing (and start after the previous append)")
        if st.is_regular:
            if np.any(dt < MIN_DT_FACTOR / st.rate):
                raise ValueError(f"Samples closer than {MIN_DT_FACTOR} / rate in a stream "
                                 f"marked regular (rate {st.rate:g})")
            base = st.n - 1 if st.t_last is not None else 0  # global 0-based index of tt[0]
            for j in np.flatnonzero(dt > GAP_FACTOR / st.rate):
                st.disc.append((base + j + 1, base + j + 2))
            first = -(-st.n // st.df) * st.df  # first coarse index >= st.n
            st.t_coarse.append(t[np.arange(first - st.n, n, st.df)])

        n_total = st.n + n
        st.g['t'].resize((1, n_total))
        st.g['Samples'].resize((len(st.names), n_total))
        st.g['t'][0, st.n:n_total] = t
        st.g['Samples'][:, st.n:n_total] = samples.T
        st.n = n_total
        st.t_last = float(t[-1])

    def close(self) -> None:
        """Write the dependent variables and top-level fields, and close the file."""
        if self._f is None:
            return
        f = self._f
        try:
            if not self._streams:
                raise ValueError("A TEA file needs at least one stream")
            for st in self._streams:
                if st.n == 0:
                    raise ValueError(f"Stream {st.rate:g} has no samples")
                self._write_stream_dependents(st)

            write_string(f, 'tea_version', '2.0')
            if self._t_units:
                write_string(f, 't_units', self._t_units)
            if self._t_offset is not None:
                write_scalar(f, 't_offset', self._t_offset)
                write_string(f, 't_offset_units', self._t_offset_units)
                write_scalar(f, 't_offset_scale', self._t_offset_scale)
            all_names = [n for st in self._streams for n in st.names]
            write_char_matrix(f, 'channel_names', all_names)
            write_array(f, 'channel_rates',
                        np.array([[st.rate for st in self._streams for _ in st.names]]).T)
            if self._hdr:
                write_hdr_fields(f, self._hdr)

            if len(self._streams) == 1:
                # Version-1 readers look for these at the top level: hard links,
                # so the data is stored once.
                g = self._streams[0].g
                for name in _V1_LINKED:
                    if name in g:
                        f[name] = g[name]
        finally:
            f.close()
            self._f = None
        write_matlab_header(self.file_path)

    @staticmethod
    def _write_stream_dependents(st: _StreamState) -> None:
        """Store the dependents gathered during `append` (regular) or recomputed (irregular)."""
        g = st.g
        if st.is_regular:
            t_coarse = np.concatenate(st.t_coarse)
            disc = np.array(st.disc, dtype=np.float64).reshape(-1, 2)
            cont = np.column_stack([np.concatenate([[1], disc[:, 1]]),
                                    np.concatenate([disc[:, 0], [st.n]])])
            _write_dependent_datasets(g, t_coarse, st.df, disc, cont, not st.disc)
        else:
            _write_dependent_datasets(g, *dependents(g['t'][0, :], st.rate, False))


def write(file_path: str | os.PathLike, streams: Sequence[Stream], *,
          t_offset: int | float | None = None, t_offset_units: str = 'posix_s',
          t_offset_scale: float = 1.0, t_units: str = 's',
          hdr: dict | None = None, compress: bool = False) -> None:
    """Write streams (one per sample rate) to a TEA schema-version-2 file.

    Example:
        tea.write('clip.mat', [
            tea.Stream(256, ['Fp1', 'Fp2'], 'uV', x256, t256),
            tea.Stream(1024, ['DC1'], 'mV', x1024, t1024),
        ], t_offset=1770000000, compress=True)
    """
    with Writer(file_path, t_offset=t_offset, t_offset_units=t_offset_units,
                t_offset_scale=t_offset_scale, t_units=t_units, hdr=hdr,
                compress=compress) as w:
        for s in streams:
            w.add_stream(s.rate, s.channel_names, s.units, is_regular=s.is_regular,
                         dtype=s.dtype, scale=s.scale, offset=s.offset)
            w.append(s.rate, s.t, s.samples)
