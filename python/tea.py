"""
tea.py - Python implementation of the TEA (Time-Encoded Array) class.

Produces MATLAB v7.3 (.mat) compatible HDF5 files that can be opened
by the MATLAB TEA class or any HDF5 reader.

Uses hdf5storage for initial file creation (MATLAB header) and h5py
for chunked/resizable dataset operations.
"""

import numpy as np
import h5py
import hdf5storage
import os


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

    def __init__(self, file_path, SR, is_regular, *,
                 t_units='', hdr=None, tea_version='1.0'):
        """
        Create or bind to a TEA file.

        Args:
            file_path: Path to the .mat file
            SR: Sample rate in Hz (None for irregular data)
            is_regular: True if samples are regularly spaced
            t_units: Units of t ('s', 'us', 'ms', etc.)
            hdr: Free-form metadata dict
            tea_version: Format version string
        """
        self._file_path = str(file_path)
        self._SR = float(SR) if SR is not None else None
        self._is_regular = bool(is_regular)
        self.t_units = str(t_units)
        self.hdr = hdr if hdr is not None else {}
        self.tea_version = str(tea_version)
        self._is_initialized = False

        # Validate SR
        if is_regular:
            if SR is None or SR <= 0:
                raise ValueError("SR must be a positive number when is_regular=True")
        else:
            if SR is not None and SR <= 0:
                raise ValueError("SR must be positive or None")

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
    def file_path(self):
        """Bound file path (immutable)."""
        return self._file_path

    @property
    def SR(self):
        """Sample rate in Hz (immutable)."""
        return self._SR

    @property
    def is_regular(self):
        """Regularity flag (immutable)."""
        return self._is_regular

    @property
    def N(self):
        """Total sample count (read from file)."""
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return 0
        with h5py.File(self._file_path, 'r') as f:
            # t stored as HDF5 (1, N) for MATLAB [N, 1]
            return f['t'].shape[1]

    @property
    def C(self):
        """Channel count (read from file)."""
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return 0
        with h5py.File(self._file_path, 'r') as f:
            # Samples stored as HDF5 (C, N) for MATLAB [N, C]
            return f['Samples'].shape[0]

    @property
    def ch_map(self):
        """Channel identity vector (read from file)."""
        if not os.path.isfile(self._file_path) or not self._is_initialized:
            return np.array([])
        with h5py.File(self._file_path, 'r') as f:
            if 'ch_map' in f:
                return np.squeeze(f['ch_map'][()]).ravel().astype(np.float64)
            else:
                return np.arange(1, self.C + 1, dtype=np.float64)

    # ============ Write ============

    def write(self, t, samples):
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

        if not self._is_initialized:
            self._create_file(t, samples)
            self._is_initialized = True
            print(f"TEA file created: {self._file_path} ({N_new} samples, {C_new} channels)")
        else:
            self._append_time(t, samples)

    # ============ Write Channels ============

    def write_channels(self, samples, ch_map=None):
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
            N_file = f['t'].shape[1]
            C_old = f['Samples'].shape[0]
            N_new, C_new = samples.shape

            if N_new != N_file:
                raise ValueError(f"New channels must have {N_file} rows, but has {N_new}")

            C_total = C_old + C_new

            # Resize and append columns
            f['Samples'].resize((C_total, N_file))
            f['Samples'][C_old:C_total, :] = samples.T

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

        print(f"TEA channels appended: {self._file_path} (now {C_total} channels)")

    # ============ Read ============

    def read(self, channels=None, t_range=None, s_range=None):
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

    def refresh(self):
        """Recompute all dependent variables from the full t vector."""
        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(f"File not found: {self._file_path}")

        with h5py.File(self._file_path, 'r+') as f:
            t = f['t'][0, :]
            self._write_dependents(f, t)
            if 'tea_version' not in f:
                self._write_h5_string(f, 'tea_version', '1.0')

        print(f"TEA refreshed: {self._file_path}")

    # ============ Info ============

    def info(self):
        """Return a summary dict of the TEA file."""
        result = {
            'file_path': self._file_path,
            'SR': self._SR,
            'is_regular': self._is_regular,
            'N': self.N,
            'C': self.C,
            'ch_map': self.ch_map,
            't_units': self.t_units,
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

    def _validate_t(self, t):
        if len(t) == 0:
            raise ValueError("t must not be empty")
        if not np.issubdtype(t.dtype, np.number):
            raise ValueError("t must be numeric")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly monotonically increasing")

    def _validate_regularity(self, t):
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

    def _create_file(self, t, samples):
        N, C = len(t), samples.shape[1]

        # Create file with MATLAB-compatible header via hdf5storage
        # Only scalar/logical metadata here; strings written via h5py for consistency
        meta = {}
        meta['SR'] = np.float64(self._SR) if self._SR is not None else np.zeros((0, 0))
        meta['isRegular'] = np.bool_(self._is_regular)

        hdf5storage.savemat(self._file_path, meta, format='7.3', oned_as='column')

        # Add chunked datasets and string metadata via h5py
        chunk_n = min(32000, max(1, N))

        with h5py.File(self._file_path, 'r+') as f:
            # t: MATLAB [N,1] → HDF5 (1, N)
            ds_t = f.create_dataset('t', data=t.reshape(1, -1),
                             maxshape=(1, None), chunks=(1, chunk_n))
            ds_t.attrs['MATLAB_class'] = np.bytes_('double')
            # Samples: MATLAB [N,C] → HDF5 (C, N)
            ds_s = f.create_dataset('Samples', data=samples.T,
                             maxshape=(None, None), chunks=(max(1, C), chunk_n))
            ds_s.attrs['MATLAB_class'] = np.bytes_('double')
            # Write string metadata via h5py (consistent row-vector format)
            self._write_h5_string(f, 'tea_version', self.tea_version)
            if self.t_units:
                self._write_h5_string(f, 't_units', self.t_units)
            # Write dependents
            self._write_dependents(f, t)

    # ============ Private: Append ============

    def _append_time(self, t, samples):
        N_new, C_new = len(t), samples.shape[1]

        with h5py.File(self._file_path, 'r+') as f:
            N_old = f['t'].shape[1]
            C_old = f['Samples'].shape[0]

            if C_new != C_old:
                raise ValueError(
                    f"Channel count mismatch: file has {C_old} but write has {C_new}. "
                    "Use write_channels to add columns.")

            t_last = float(f['t'][0, N_old - 1])
            if t[0] <= t_last:
                raise ValueError(
                    f"New t must start after existing t. Last: {t_last}, first new: {t[0]}")

            N_total = N_old + N_new

            # Resize and append
            f['t'].resize((1, N_total))
            f['t'][0, N_old:N_total] = t

            f['Samples'].resize((C_old, N_total))
            f['Samples'][:, N_old:N_total] = samples.T

            # Update dependents incrementally
            self._update_dependents_incremental(f, t, t_last, N_old, N_total)

        print(f"TEA file appended: {self._file_path} (now {N_total} samples)")

    # ============ Private: Dependents ============

    def _write_dependents(self, f, t):
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

    def _update_dependents_incremental(self, f, t_new, t_last, N_old, N_total):
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

    def _resolve_channels(self, f, channels, total_C):
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

    def _find_range_from_time(self, f, t_range, total_N):
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

    def _compute_disc_info(self, t_out):
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
    def _write_h5_scalar(f, name, value, logical=False):
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
    def _write_h5_array(f, name, data):
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
    def _write_h5_string(f, name, value):
        """Write a string as a MATLAB-compatible HDF5 dataset (row vector)."""
        if name in f:
            del f[name]
        # Store as uint16 column in HDF5 (len, 1) → MATLAB sees [1, len] row vector
        encoded = np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)  # shape (len, 1)
        ds = f.create_dataset(name, data=encoded)
        ds.attrs['MATLAB_class'] = np.bytes_('char')
        ds.attrs['MATLAB_int_decode'] = np.int32(2)
