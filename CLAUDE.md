# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dual-language** (MATLAB + Python) timeseries I/O library for neurophysiology. `TimeEncodedArray` (TEA) stores multichannel sampled data with explicit per-sample timestamps in HDF5-backed `.mat` (v7.3) files. Supports regular (constant `dt`) and irregular sampling, natural discontinuity handling, and efficient partial reads. Cross-language determinism is verified by paired `write_test_cases` / `compare_test_cases` scripts in both languages.

## Running

**MATLAB:**
```matlab
addpath('matlab')
tea = TEA('file.mat', SR, isRegular, 't_units','s', 'expected_length', N, 'compress', true);
tea.write(t, Samples);              % append
tea.finalize();                     % trim preallocated datasets
[data, t, info] = tea.read(channels, t_range, s_range);
runtests('test_tea')                % 38 tests
runtests('test_tea_v2')             % 8 tests (schema v2; reads files from python/tests/write_v2_test_cases.py)
s = tea.stream(rate); s.read(...)   % schema v2: one TEAStream per sample rate
tea = TEA('f.mat', SR, true, 'scale', g, 'offset', o); tea.write(t, int32(counts));  % counts + scale
Data = tea.read([], [], [], 'DataType', 'int');   % '' = physical double when scaled, 'single', 'int'
```

**Python:**
```python
from tea import TEA                 # from python/
tea = TEA('file.mat', SR, is_regular=True, ...)
tea.write(t, samples)
tea.finalize()
data, t, info = tea.read(channels=..., t_range=..., s_range=...)
# Schema v2 (several rates per file):
tea.write(path, [tea.Stream(rate, names, units, samples, t, dtype=np.int32, scale=g, offset=o), ...], t_offset=...)  # or tea.Writer
f = tea.read(path); f.rates; f.stream(rate).read(..., dtype=None | np.float32 | 'int')
# Tests: uv run pytest -q   (65 tests: test_tea.py 41, test_tea_v2.py 24)
# Install: pip install -e .         (from repo root, uses pyproject.toml)
```

Storage classes (0.3.0): `dtype` float64 (default), float32, int16, int32, int64; `scale`/`offset`
(per channel) only with an integer dtype, physical = counts * scale + offset; readers return physical
float64 by default when `scale` exists, else as stored; `dtype='int'` gives the counts.

Cross-validation: run `write_test_cases` in one language, `compare_test_cases` in the other.

## Directory Layout

- `matlab/TEA.m` — class (~1000 lines); `matlab/TEAStream.m` — read-only view of one rate (schema v2).
- `matlab/tests/test_tea.m`, `test_tea_v2.m`, `write_test_cases.m`, `compare_test_cases.m`.
- `matlab/examples/example_brew_and_drink.m`.
- `python/tea.py` — module (~1500 lines); `python/tea_h5.py` — MATLAB-side HDF5 helpers (datasets with `MATLAB_class`, v7.3 header).
- `python/tests/test_tea.py`, `test_tea_v2.py`, `write_test_cases.py`, `compare_test_cases.py`, `write_v2_test_cases.py`.
- `python/examples/example_tea.py`.
- `schema/tea_schema.md` — full file-format specification (schema versions 1 and 2; v2 = one group per rate under `/rates`, one-rate v2 files also carry the v1 fields as hard links).
- `cross_validation_data/` — artifacts from cross-language tests.
- `pyproject.toml`, `LICENSE` (MIT), `README.md`.

## Public API (symmetric across MATLAB / Python)

- **Constructor:** `TEA(file_path, SR, isRegular, varargin)` / `TEA(file_path, SR, is_regular, **kwargs)`.
  Optional: `t_units`, `t_offset`, `t_offset_units`, `t_offset_scale`, `hdr` (free-form metadata struct/dict), `compress` (DEFLATE level 1; shuffle for integer classes only), `expected_length`, `expected_channels` (pre-allocation), `scale`, `offset` (integer counts only; Python also `dtype`).
- `write(t, Samples)` — append; enforces monotonicity (`t(1) > t_last`). MATLAB stores `class(Samples)`; Python converts to the file's `dtype`.
- `write_absolute(t_abs, Samples)` — append with absolute timestamps.
- `write_channels(Samples, ch_map)` — add new channel columns (refused on a scaled file).
- `finalize()` — trim pre-allocated HDF5 datasets to actual written size.
- `read(channels, t_range, s_range, data_var_name, 'DataType', dt)` / `read(channels, t_range, s_range, dtype=)` — partial read; physical by default when scaled, counts on request.
- `refresh()`, `info()` (includes `dtype`, `scaled`).
- Properties: immutable `file_path`, `SR`, `isRegular`, `compress`; mutable `t_units`, `t_offset`, `t_offset_units`, `t_offset_scale`, `hdr`, `tea_version`; dependent `N` (samples), `C` (channels), `ch_map`; `dtype`, `scale`, `offset`.

## File Format Summary

- Required HDF5 datasets: `t`, `Samples`, `SR`, `isRegular`.
- Computed dependents: `t_coarse`, `df_t_coarse` (coarse decimated index, ~1 sample/sec, for fast time lookups), `isContinuous`, `cont` (continuous-block boundaries), `disc` (gap boundaries).
- Optional: `t_units`, `t_offset`, `ch_map`, `hdr`, `scale`/`offset` (`[1 x C]`, integer `Samples` only).
- Absolute time model: `t_abs = t_offset * t_offset_scale + t`.
- `Samples` class: `double`, `single`, `int16`, `int32`, `int64`; integer = counts, physical = `counts .* scale + offset`.

## Gotchas

- **Discontinuity threshold:** gaps detected when `diff(t) > 1.1/SR` (11 % tolerance); within a segment `0.9/SR <= dt <= 1.1/SR` (closer samples are refused).
- **Compression filters:** DEFLATE 1; shuffle only for integer `Samples` (25-28 % smaller on int32 EEG counts, 1.6x larger on float64). `t` is never shuffled.
- **Float64 precision warning** triggers if ULP at `t(1)` exceeds 1 % of the sample spacing — use `t_offset` for long recordings to keep `t` near zero.
- **`SR` is samples per `t_units`**, not always Hz. On file creation, a warning is issued if `t_units != 's'`.
- **Pre-allocation is visible internally:** `N_written`/`C_written` (logical) vs `N_allocated`/`C_allocated` (physical HDF5 size). Call `finalize()` before shipping files to trim unused space.
- **Monotonicity is strict** — appending `t(1) <= t_last` raises an error; no silent sort.
- **Python deps:** `numpy`, `h5py` (declared in `pyproject.toml`). No `hdf5storage` dependency anymore; `h5py` writes v7.3-compatible `.mat` directly.
- **MATLAB deps:** none beyond built-in `matfile` / HDF5. No toolbox required.

## Related Repositories

Sibling of `sine_and_paired_analysis`, `nrd2mat`, `nsd2mat`, `med2mat`, `pth`, `sine_stim_xls_processing`. Standalone library — not tightly coupled to any of them, though the file-format philosophy matches what `nrd2mat`/`nsd2mat` emit (explicit `t`, `Samples`, `SR`, coarse index).
