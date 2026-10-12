"""
tea_h5.py - MATLAB v7.3 HDF5 helpers for the TEA format.

Small functions that write and read MATLAB-compatible datasets (scalars,
arrays, strings, char matrices, `hdr` structs) with the `MATLAB_class`
attributes MATLAB's `load` and `h5read` expect, and the 512-byte userblock
header that makes an HDF5 file a `.mat` v7.3 file. Shapes are given as MATLAB
sees them; HDF5 stores the transpose.
"""

from __future__ import annotations

import time
from typing import Sequence

import h5py
import numpy as np
import numpy.typing as npt


def write_scalar(g: h5py.Group, name: str, value: int | float,
                 logical: bool = False) -> None:
    """Write a scalar as a MATLAB-compatible dataset (shape [1 x 1])."""
    if name in g:
        del g[name]
    if logical:
        ds = g.create_dataset(name, data=np.array([[np.uint8(1 if value else 0)]], dtype=np.uint8))
        ds.attrs['MATLAB_class'] = np.bytes_('logical')
        ds.attrs['MATLAB_int_decode'] = np.int32(1)
    else:
        ds = g.create_dataset(name, data=np.array([[np.float64(value)]]))
        ds.attrs['MATLAB_class'] = np.bytes_('double')


def write_array(g: h5py.Group, name: str, data: npt.ArrayLike) -> None:
    """Write a double array in MATLAB-logical shape.

    The caller gives e.g. (1, N) for a row vector; it is stored as HDF5 (N, 1)
    so MATLAB sees [1 x N].
    """
    if name in g:
        del g[name]
    arr = np.asarray(data, dtype=np.float64)
    ds = g.create_dataset(name, data=arr.T)
    ds.attrs['MATLAB_class'] = np.bytes_('double')


def write_string(g: h5py.Group, name: str, value: str) -> None:
    """Write a string as a MATLAB char row vector."""
    if name in g:
        del g[name]
    # uint16 column (len, 1) in HDF5 -> MATLAB [1 x len]
    encoded = np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)
    ds = g.create_dataset(name, data=encoded)
    ds.attrs['MATLAB_class'] = np.bytes_('char')
    ds.attrs['MATLAB_int_decode'] = np.int32(2)


def write_char_matrix(g: h5py.Group, name: str, strings: Sequence[str]) -> None:
    """Write strings as a MATLAB char matrix, one row per string, space padded.

    MATLAB reads it as a [len(strings) x max_len] char array; cellstr() gives
    the cell array of names.
    """
    if name in g:
        del g[name]
    max_len = max([1] + [len(s) for s in strings])
    arr = np.full((len(strings), max_len), ord(' '), dtype=np.uint16)
    for i, s in enumerate(strings):
        arr[i, :len(s)] = [ord(c) for c in s]
    ds = g.create_dataset(name, data=arr.T)  # MATLAB [n, len] stored as HDF5 (len, n)
    ds.attrs['MATLAB_class'] = np.bytes_('char')
    ds.attrs['MATLAB_int_decode'] = np.int32(2)


def read_char_matrix(ds: h5py.Dataset) -> list[str] | None:
    """Rows of a MATLAB char matrix, trailing spaces and NULs removed.

    Returns None when the dataset is not a char array (e.g. a MATLAB cell).
    """
    data = ds[()]
    if not np.issubdtype(np.asarray(data).dtype, np.integer):
        return None
    rows = np.atleast_2d(np.asarray(data)).T  # HDF5 (len, n) -> (n, len)
    return [''.join(chr(c) for c in row).rstrip(' \x00') for row in rows]


def read_scalar(g: h5py.Group, name: str, default=None):
    """A MATLAB scalar (or `default` when absent or stored empty)."""
    if name not in g:
        return default
    ds = g[name]
    if ds.attrs.get('MATLAB_empty', 0):
        return default
    return np.squeeze(ds[()]).item()


def read_string(g: h5py.Group, name: str, default: str = '') -> str:
    """A MATLAB char row vector as a str (`default` when absent)."""
    if name not in g:
        return default
    rows = read_char_matrix(g[name])
    return ''.join(rows) if rows is not None else default


def write_hdr_fields(g: h5py.Group, hdr_dict: dict) -> None:
    """Write a dict as the MATLAB struct group `hdr`.

    Each key becomes a field of the struct:
      - list[str]  -> 2D char matrix (use cellstr() in MATLAB)
      - str        -> char row vector
      - int/float  -> double scalar
      - np.ndarray -> double array
    """
    if 'hdr' in g:
        del g['hdr']
    hdr_grp = g.create_group('hdr')
    hdr_grp.attrs['MATLAB_class'] = np.bytes_('struct')

    for key, val in hdr_dict.items():
        if isinstance(val, list) and len(val) > 0 and all(isinstance(v, str) for v in val):
            # List of strings -> 2D char matrix (MATLAB char array), NUL padded
            max_len = max(len(s) for s in val)
            char_arr = np.zeros((len(val), max_len), dtype=np.uint16)
            for i, s in enumerate(val):
                for j, ch in enumerate(s):
                    char_arr[i, j] = ord(ch)
            ds = hdr_grp.create_dataset(key, data=char_arr.T)  # MATLAB (N, M) as HDF5 (M, N)
            ds.attrs['MATLAB_class'] = np.bytes_('char')
            ds.attrs['MATLAB_int_decode'] = np.int32(2)
        elif isinstance(val, str):
            write_string(hdr_grp, key, val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            write_scalar(hdr_grp, key, val)
        elif isinstance(val, np.ndarray):
            arr = val.astype(np.float64)
            stored = arr.T if arr.ndim > 1 else arr.reshape(1, -1)
            ds = hdr_grp.create_dataset(key, data=stored)
            ds.attrs['MATLAB_class'] = np.bytes_('double')


def write_matlab_header(file_path: str) -> None:
    """Write the 128-byte MATLAB v7.3 MAT-file header into the file's userblock.

    The file must have been created with `userblock_size=512`.
    """
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


def schema_version_of(f: h5py.File) -> int:
    """Schema version stored in an open file; 1 when the file has none."""
    if 'schema_version' in f:
        return int(round(float(np.squeeze(f['schema_version'][()]))))
    return 1
