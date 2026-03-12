"""
tea.py - Python implementation of the TEA (Time-Encoded Array) class.

Produces MATLAB v7.3 (.mat) compatible HDF5 files that can be opened
by the MATLAB TEA class or any HDF5 reader.

Uses h5py for all HDF5 operations. The MATLAB v7.3 userblock header
is written directly as raw bytes.
"""

from __future__ import annotations

import logging
import os
import time
import warnings

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Type aliases
ArrayFloat64 = npt.NDArray[np.float64]


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
                 expected_channels: int | None = None) -> None:
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
            compress: If True, use DEFLATE level 1 + shuffle on t and Samples
            expected_length: Pre-allocate N samples (avoids resize on append)
            expected_channels: Pre-allocate C channels (avoids resize on write_channels)
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

    # ============ Properties ============

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
            else:
                return np.arange(1, self.C + 1, dtype=np.float64)

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
        t = np.asarray(t, dtype=np.float64).ravel()
        samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
        if samples.ndim == 2 and samples.shape[0] == 1 and len(t) > 1:
            samples = samples.T

        N_new = len(t)
        self._validate_t(t)
        if samples.shape[0] != N_new:
            raise ValueError(
                f"Samples must have {N_new} rows to match t, but has {samples.shape[0]}")
        C_new = samples.shape[1]

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

        samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))

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

            self._write_h5_array(f, 'ch_map', new_map.reshape(1, -1))  # MATLAB [1, C]

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
             s_range: tuple[int, int] | None = None,
             ) -> tuple[ArrayFloat64, ArrayFloat64, dict]:
        """
        Read data from the TEA file.

        Args:
            channels: Channel numbers (uses ch_map). None for all.
            t_range: (start, end) in t units. None to skip.
            s_range: (start, end) sample indices, 0-indexed inclusive. None to skip.

        Returns:
            (Data, t_out, disc_info) tuple.
            Data: (N, C) array. t_out: 1D array. disc_info: dict.
        """
        if t_range is not None and s_range is not None:
            raise ValueError("Specify either t_range or s_range, not both.")

        with h5py.File(self._file_path, 'r') as f:
            total_N = f['t'].shape[1]
            total_C = f['Samples'].shape[0]

            # Channel selection
            channel_indices = self._resolve_channels(f, channels, total_C)

            # Determine sample range
            if s_range is not None:
                s_start, s_end = int(s_range[0]), int(s_range[1])
                if s_start < 0 or s_end >= total_N or s_start > s_end:
                    raise ValueError(
                        f"Invalid s_range [{s_start}, {s_end}]. Valid: [0, {total_N - 1}]")
            elif t_range is not None:
                s_start, s_end = self._find_range_from_time(f, t_range, total_N)
            else:
                s_start, s_end = 0, total_N - 1

            # Empty result
            n_ch = len(channel_indices) if channel_indices is not None else total_C
            if total_N == 0 or s_start > s_end or s_start < 0:
                empty_disc = {'is_discontinuous': False,
                              'cont': np.zeros((0, 2)), 'disc': np.zeros((0, 2))}
                return np.zeros((0, n_ch)), np.zeros(0), empty_disc

            # Read t
            t_out = f['t'][0, s_start:s_end + 1]

            # Read Samples (HDF5 shape: (C, N) → transpose to (N, C))
            if channel_indices is not None:
                Data = f['Samples'][channel_indices, s_start:s_end + 1].T
            else:
                Data = f['Samples'][:, s_start:s_end + 1].T

            disc_info = self._compute_disc_info(t_out)
            return Data, t_out, disc_info

    # ============ Refresh ============

    def refresh(self) -> None:
        """Recompute all dependent variables from the full t vector."""
        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(f"File not found: {self._file_path}")

        with h5py.File(self._file_path, 'r+') as f:
            t = f['t'][0, :]
            self._write_dependents(f, t)
            if 'tea_version' not in f:
                self._write_h5_string(f, 'tea_version', '1.0')

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
        }

        if os.path.isfile(self._file_path):
            with h5py.File(self._file_path, 'r') as f:
                if 'isContinuous' in f:
                    result['isContinuous'] = bool(np.squeeze(f['isContinuous'][()]))
                N = result['N']
                if N > 0:
                    result['t_first'] = float(f['t'][0, 0])
                    result['t_last'] = float(f['t'][0, N - 1])

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
        if len(t) < 2:
            return
        dt = np.diff(t)
        expected_dt = 1.0 / self._SR
        gap_threshold = 1.1 * expected_dt
        regular_mask = dt <= gap_threshold
        regular_fraction = np.sum(regular_mask) / len(dt)

        if regular_fraction < 0.5:
            raise ValueError(
                f"is_regular=True but only {regular_fraction * 100:.0f}% of dt values are within 1.1/SR")

        if np.any(regular_mask):
            regular_dt = dt[regular_mask]
            max_dev = np.max(np.abs(regular_dt - expected_dt)) / expected_dt
            if max_dev > 0.1:
                raise ValueError(
                    f"is_regular=True but within-segment dt deviates from 1/SR by {max_dev * 100:.1f}%")

    # ============ Private: File Creation ============

    def _create_file(self, t: ArrayFloat64, samples: ArrayFloat64) -> None:
        N, C = len(t), samples.shape[1]

        # Determine allocation sizes
        N_alloc = max(N, self._expected_length or N)
        C_alloc = max(C, self._expected_channels or C)
        chunk_n = min(32000, max(1, N_alloc))
        chunk_c = max(1, C_alloc)

        # Compression kwargs for t and Samples datasets
        comp_kwargs = {}
        if self._compress:
            comp_kwargs = {'compression': 'gzip', 'compression_opts': 1,
                           'shuffle': True}

        # Create HDF5 file with 512-byte userblock for MATLAB v7.3 compatibility
        with h5py.File(self._file_path, 'w', userblock_size=512) as f:
            # SR: scalar for regular, empty [] for irregular
            if self._SR is not None:
                self._write_h5_scalar(f, 'SR', np.float64(self._SR))
            else:
                ds_sr = f.create_dataset('SR', data=np.array([0, 0], dtype=np.uint64))
                ds_sr.attrs['MATLAB_class'] = np.bytes_('double')
                ds_sr.attrs['MATLAB_empty'] = np.uint8(1)
            # isRegular
            self._write_h5_scalar(f, 'isRegular', self._is_regular, logical=True)
            # t: HDF5 (1, N_alloc), write first N values
            ds_t = f.create_dataset('t', shape=(1, N_alloc),
                             maxshape=(1, None), chunks=(1, chunk_n),
                             dtype=np.float64, **comp_kwargs)
            ds_t.attrs['MATLAB_class'] = np.bytes_('double')
            ds_t[0, :N] = t
            # Samples: HDF5 (C_alloc, N_alloc), write first (C, N) values
            ds_s = f.create_dataset('Samples', shape=(C_alloc, N_alloc),
                             maxshape=(None, None), chunks=(chunk_c, chunk_n),
                             dtype=np.float64, **comp_kwargs)
            ds_s.attrs['MATLAB_class'] = np.bytes_('double')
            ds_s[:C, :N] = samples.T
            # String metadata
            self._write_h5_string(f, 'tea_version', self.tea_version)
            if self.t_units:
                self._write_h5_string(f, 't_units', self.t_units)
            # t_offset metadata
            if self._t_offset is not None:
                self._write_h5_scalar(f, 't_offset', self._t_offset)
                self._write_h5_string(f, 't_offset_units', self._t_offset_units)
                self._write_h5_scalar(f, 't_offset_scale', float(self._t_offset_scale))
            # Dependent variables
            self._write_dependents(f, t)

            # hdr: free-form metadata stored as MATLAB struct group
            if self.hdr:
                self._write_hdr_fields(f, self.hdr)

        # Write the 128-byte MATLAB v7.3 header into the userblock
        self._write_matlab_header(self._file_path)

        # Track logical and physical sizes
        self._N_written = N
        self._C_written = C
        self._N_allocated = N_alloc
        self._C_allocated = C_alloc

    @staticmethod
    def _write_matlab_header(file_path: str) -> None:
        """Write the 128-byte MATLAB v7.3 MAT-file header into the userblock."""
        desc = (f"MATLAB 7.3 MAT-file, Platform: Python, "
                f"Created on: {time.strftime('%c')} "
                f"HDF5 schema 1.00 .")
        # Bytes 1-116: description (space-padded, ASCII)
        desc_bytes = desc.encode('ascii')[:116].ljust(116, b' ')
        # Bytes 117-124: subsys data offset (8 zero bytes)
        # Bytes 125-126: version [0x00, 0x02]
        # Bytes 127-128: endian indicator 'IM' (little-endian)
        trailer = b'\x00' * 8 + b'\x00\x02' + b'IM'
        header = desc_bytes + trailer  # 128 bytes total

        with open(file_path, 'r+b') as fid:
            fid.write(header)

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
        N = len(t)

        # df_t_coarse
        if self._is_regular and self._SR is not None and self._SR > 0:
            df = max(1, round(self._SR))
        else:
            if N > 0 and (t[-1] - t[0]) > 0:
                df = max(1, round(N / (t[-1] - t[0])))
            else:
                df = 1

        # t_coarse
        t_coarse = t[::df] if N > 0 else np.array([])
        # t_coarse: MATLAB [M, 1] column vector
        tc_data = t_coarse.reshape(-1, 1) if len(t_coarse) > 0 else np.zeros((0, 1))
        self._write_h5_array(f, 't_coarse', tc_data)
        self._write_h5_scalar(f, 'df_t_coarse', float(df))

        # isContinuous, cont, disc
        is_cont = True
        if self._is_regular and self._SR is not None and self._SR > 0 and N > 1:
            dt = np.diff(t)
            gap_threshold = 1.1 / self._SR
            gap_mask = dt > gap_threshold

            if np.any(gap_mask):
                is_cont = False
                gap_idx = np.where(gap_mask)[0]  # 0-indexed
                # Store as 1-indexed for MATLAB compatibility
                disc = np.column_stack([gap_idx + 1, gap_idx + 2])  # [last_before, first_after] 1-indexed
                starts = np.concatenate([[1], gap_idx + 2])
                ends = np.concatenate([gap_idx + 1, [N]])
                cont = np.column_stack([starts, ends])
                # Pass in MATLAB shape [n, 2]; helper handles HDF5 transposition
                self._write_h5_array(f, 'disc', disc.astype(np.float64))
                self._write_h5_array(f, 'cont', cont.astype(np.float64))

        if is_cont:
            # Remove disc/cont if they exist (was continuous)
            for var in ['disc', 'cont']:
                if var in f:
                    del f[var]

        self._write_h5_scalar(f, 'isContinuous', is_cont, logical=True)

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
            gap_threshold = 1.1 / self._SR

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
                self._write_h5_scalar(f, 'isContinuous', True, logical=True)
                for var in ['disc', 'cont']:
                    if var in f:
                        del f[var]
            else:
                self._write_h5_scalar(f, 'isContinuous', False, logical=True)
                self._write_h5_array(f, 'disc', all_disc)  # MATLAB [n, 2]
                starts = np.concatenate([[1], all_disc[:, 1]])
                ends = np.concatenate([all_disc[:, 0], [N_total]])
                cont = np.column_stack([starts, ends])
                self._write_h5_array(f, 'cont', cont)  # MATLAB [n, 2]
        else:
            self._write_h5_scalar(f, 'isContinuous', True, logical=True)

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
            self._write_h5_array(f, 't_coarse', combined_tc.reshape(-1, 1))  # MATLAB [M, 1]

    # ============ Private: Read Helpers ============

    def _resolve_channels(self, f: h5py.File, channels: list[int] | None,
                          total_C: int) -> np.ndarray | None:
        if channels is None:
            return None
        if 'ch_map' in f:
            file_ch_map = np.squeeze(f['ch_map'][()]).ravel()
        else:
            file_ch_map = np.arange(1, total_C + 1)

        indices = []
        for ch in channels:
            found = np.where(file_ch_map == ch)[0]
            if len(found) == 0:
                raise ValueError(f"Channel {ch} not found in ch_map")
            indices.append(found[0])
        return np.array(indices)

    def _find_range_from_time(self, f: h5py.File,
                              t_range: tuple[float, float],
                              total_N: int) -> tuple[int, int]:
        if total_N == 0:
            return 0, -1

        t_start_req, t_end_req = t_range
        t_first = float(f['t'][0, 0])
        t_last = float(f['t'][0, total_N - 1])

        if t_start_req > t_last or t_end_req < t_first:
            return 0, -1  # out of range

        t_start = max(t_start_req, t_first)
        t_end = min(t_end_req, t_last)

        has_tc = 't_coarse' in f and 'df_t_coarse' in f
        if not has_tc:
            # Full t load fallback
            full_t = f['t'][0, :]
            s_start = int(np.searchsorted(full_t, t_start, side='left'))
            s_end = int(np.searchsorted(full_t, t_end, side='right') - 1)
            return s_start, s_end

        # Use t_coarse for fast lookup
        tc = f['t_coarse'][()].ravel()
        df = int(np.squeeze(f['df_t_coarse'][()]))

        # Find start
        if t_start <= t_first:
            s_start = 0
        else:
            d_idx = np.searchsorted(tc, t_start, side='left')
            ws = max(0, (d_idx - 1) * df)
            we = min((d_idx + 1) * df, total_N)
            tw = f['t'][0, ws:we]
            ri = np.searchsorted(tw, t_start, side='left')
            s_start = ws + ri

        # Find end
        if t_end >= t_last:
            s_end = total_N - 1
        else:
            d_idx = np.searchsorted(tc, t_end, side='right')
            ws = max(0, (d_idx - 1) * df)
            we = min((d_idx + 1) * df, total_N)
            tw = f['t'][0, ws:we]
            ri = np.searchsorted(tw, t_end, side='right') - 1
            s_end = ws + max(0, ri)

        s_start = max(0, min(s_start, total_N - 1))
        s_end = max(0, min(s_end, total_N - 1))
        return int(s_start), int(s_end)

    def _compute_disc_info(self, t_out: ArrayFloat64) -> dict:
        NN = len(t_out)
        result = {'is_discontinuous': False,
                  'cont': np.zeros((0, 2)),
                  'disc': np.zeros((0, 2))}

        if NN < 2 or not self._is_regular:
            if NN > 0:
                result['cont'] = np.array([[0, NN - 1]])
            return result

        if self._SR is None or self._SR <= 0:
            result['cont'] = np.array([[0, NN - 1]])
            return result

        dt = np.diff(t_out)
        gap_mask = dt > 1.1 / self._SR

        if np.any(gap_mask):
            result['is_discontinuous'] = True
            gi = np.where(gap_mask)[0]
            result['disc'] = np.column_stack([gi, gi + 1])
            starts = np.concatenate([[0], gi + 1])
            ends = np.concatenate([gi, [NN - 1]])
            result['cont'] = np.column_stack([starts, ends])
        else:
            result['cont'] = np.array([[0, NN - 1]])

        return result

    # ============ Private: HDF5 Helpers ============

    @staticmethod
    def _write_h5_scalar(f: h5py.File, name: str,
                         value: int | float, logical: bool = False) -> None:
        """Write a scalar value as a MATLAB-compatible HDF5 dataset (shape [1,1])."""
        if name in f:
            del f[name]
        if logical:
            ds = f.create_dataset(name, data=np.array([[np.uint8(1 if value else 0)]], dtype=np.uint8))
            ds.attrs['MATLAB_class'] = np.bytes_('logical')
            ds.attrs['MATLAB_int_decode'] = np.int32(1)
        else:
            ds = f.create_dataset(name, data=np.array([[np.float64(value)]]))
            ds.attrs['MATLAB_class'] = np.bytes_('double')

    @staticmethod
    def _write_h5_array(f: h5py.File, name: str, data: npt.ArrayLike) -> None:
        """Write a numeric array as a MATLAB-compatible HDF5 dataset.

        Data is transposed for HDF5 storage so MATLAB reads correct dimensions.
        Caller provides data in MATLAB-logical shape (e.g. (1, N) for row vector);
        stored as HDF5 (N, 1) so MATLAB sees [1, N].
        """
        if name in f:
            del f[name]
        arr = np.asarray(data, dtype=np.float64)
        ds = f.create_dataset(name, data=arr.T)
        ds.attrs['MATLAB_class'] = np.bytes_('double')

    @staticmethod
    def _write_h5_string(f: h5py.File, name: str, value: str) -> None:
        """Write a string as a MATLAB-compatible HDF5 dataset (row vector)."""
        if name in f:
            del f[name]
        # Store as uint16 column in HDF5 (len, 1) → MATLAB sees [1, len] row vector
        encoded = np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)  # shape (len, 1)
        ds = f.create_dataset(name, data=encoded)
        ds.attrs['MATLAB_class'] = np.bytes_('char')
        ds.attrs['MATLAB_int_decode'] = np.int32(2)

    @staticmethod
    def _write_hdr_fields(f: h5py.File, hdr_dict: dict) -> None:
        """Write hdr dict as a MATLAB-compatible struct group.

        Creates an HDF5 group '/hdr' with MATLAB_class='struct'.
        Each dict key becomes a field in the struct:
          - list[str]  → 2D char matrix (use cellstr() in MATLAB)
          - str        → char row vector
          - int/float  → double scalar
          - np.ndarray → double array
        """
        if 'hdr' in f:
            del f['hdr']
        hdr_grp = f.create_group('hdr')
        hdr_grp.attrs['MATLAB_class'] = np.bytes_('struct')

        for key, val in hdr_dict.items():
            if isinstance(val, list) and len(val) > 0 and all(isinstance(v, str) for v in val):
                # List of strings → 2D char matrix (MATLAB char array)
                # MATLAB: cellstr(hdr.field) to get cell array of strings
                max_len = max(len(s) for s in val)
                char_arr = np.zeros((len(val), max_len), dtype=np.uint16)
                for i, s in enumerate(val):
                    for j, ch in enumerate(s):
                        char_arr[i, j] = ord(ch)
                # Transpose: MATLAB (N, M) stored as HDF5 (M, N)
                ds = hdr_grp.create_dataset(key, data=char_arr.T)
                ds.attrs['MATLAB_class'] = np.bytes_('char')
                ds.attrs['MATLAB_int_decode'] = np.int32(2)
            elif isinstance(val, str):
                encoded = np.array([ord(c) for c in val], dtype=np.uint16).reshape(-1, 1)
                ds = hdr_grp.create_dataset(key, data=encoded)
                ds.attrs['MATLAB_class'] = np.bytes_('char')
                ds.attrs['MATLAB_int_decode'] = np.int32(2)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                ds = hdr_grp.create_dataset(key, data=np.array([[np.float64(val)]]))
                ds.attrs['MATLAB_class'] = np.bytes_('double')
            elif isinstance(val, np.ndarray):
                arr = val.astype(np.float64)
                stored = arr.T if arr.ndim > 1 else arr.reshape(1, -1)
                ds = hdr_grp.create_dataset(key, data=stored)
                ds.attrs['MATLAB_class'] = np.bytes_('double')
