# TEA: Time-Encoded Array

A standardized timeseries data format built on HDF5/`.mat` v7.3 files. Designed for multichannel sampled data with explicit per-sample timestamps, natural discontinuity handling, and efficient partial reads.

Implementations in **Python** and **MATLAB** produce identical files — verified by cross-language tests.

## Python

### Dependencies

```
numpy, h5py
```

### Quick Start

```python
from tea import TEA

# --- Create and write ---
tea = TEA('data.mat', 1000, True, t_units='s')
tea.write(t, samples)                # creates file on first call

# --- Append ---
tea.write(t2, samples2)              # appends in time
tea.write_channels(new_data, ch_map=[5, 6])  # appends channels

# --- Read ---
Data, t_out, disc_info = tea.read(channels=[1, 3], t_range=(1.0, 2.0))

# --- Info ---
s = tea.info()   # 'N', 'C', 'SR', 'isContinuous', ...

# --- Re-open existing file ---
tea2 = TEA('data.mat', 1000, True)   # validates SR/is_regular match
tea2.write(t3, s3)                   # appends
```

### Several sample rates in one file (schema version 2)

```python
import tea

# --- Write: one Stream per sample rate; t in seconds relative to t_offset, gaps allowed ---
tea.write('clip.mat', [
    tea.Stream(256,  ['Fp1', 'Fp2', 'Cz'], 'uV', x256, t256),    # samples (n, ch) float64, t (n,)
    tea.Stream(1024, ['DC1', 'DC2'], ['mV', 'uV'], x1024, t1024),
], t_offset=1770000000, compress=True)

# --- Or in blocks (large recordings) ---
with tea.Writer('clip.mat', t_offset=1770000000) as w:
    w.add_stream(256, names256, 'uV')
    w.add_stream(1024, names1024, 'uV')
    for t, x in blocks:
        w.append(256, t, x)

# --- Read (version 1 or 2) ---
f = tea.read('clip.mat')
f.rates                                    # [256.0, 1024.0]
s = f.stream(1024)                         # s.channel_names, s.units, s.N, s.C
data, t, disc = s.read(channels=['DC2'], t_range=(10.0, 12.0))

TEA('clip.mat', 256, True).stream(1024)    # same, from the version-1 class
```

A file written with one stream also carries the version-1 fields, so `TEA(path, SR, True).read()`
(and older readers) work on it unchanged. A version-1 file reads through `tea.read` as one stream.

### Running Tests

```bash
uv run pytest -q                                 # 54 tests: test_tea.py (39) and test_tea_v2.py (15)
```

## MATLAB

### Quick Start

```matlab
addpath('matlab');

% --- Create and write ---
tea = TEA('data.mat', 1000, true, 't_units', 's');
tea.write(t, Samples);                % creates file on first call

% --- Append ---
tea.write(t2, Samples2);             % appends in time
tea.write_channels(new_data, [5 6]); % appends channels

% --- Read ---
[Data, t_out, disc_info] = tea.read([1 3], [1.0, 2.0], []);

% --- Info ---
s = tea.info();   % .N, .C, .SR, .isContinuous, ...

% --- Re-open existing file ---
tea2 = TEA('data.mat', 1000, true);  % validates SR/isRegular match
tea2.write(t3, s3);                  % appends
```

### Several sample rates (schema version 2)

```matlab
tea = TEA('clip.mat', 256, true);    % any rate in the file
tea.rates                            % [256 1024]
s = tea.stream(1024);                % TEAStream: s.channel_names, s.units, s.N, s.C
[Data, t, disc_info] = s.read({'DC2'}, [10.0, 12.0], []);
```

Version-2 files are written from Python. `load('clip.mat')` gives `S.rates.r1.Samples`, `S.rates.r1.t`, ...

### Running Tests

```matlab
cd matlab/tests
runtests('test_tea')                 % 37 unit tests
runtests('test_tea_v2')              % 6 tests; uses files from python/tests/write_v2_test_cases.py
```

## Cross-Language Validation

Both implementations write 8 deterministic test cases. Two comparator scripts verify every variable matches:

```bash
# Write test files
python python/tests/write_test_cases.py
matlab> write_test_cases

# Compare from MATLAB's perspective
matlab> compare_test_cases

# Compare from Python's perspective
python python/tests/compare_test_cases.py
```

## Schema

See [schema/tea_schema.md](schema/tea_schema.md) for the full specification. The tables below are
the version-1 layout, which is also the layout of each rate group (`/rates/r1`, `/rates/r2`, ...)
in a version-2 file.

### Required

| Variable | Shape | Description |
|----------|-------|-------------|
| `t` | `[N×1]` | Timestamps (monotonically increasing) |
| `Samples` | `[N×C]` | Data matrix (any numeric type) |
| `SR` | scalar/`[]` | Sample rate in samples per `t_units` (= Hz when `t_units='s'`) |
| `isRegular` | scalar | Regular sampling flag |

### Dependent (auto-computed)

`t_coarse`, `df_t_coarse`, `isContinuous`, `cont`, `disc`

### Optional

`t_units`, `t_offset`, `t_offset_units`, `t_offset_scale`, `ch_map`, `SR_original`, `hdr`, `tea_version`

## API

| Method | Description |
|--------|-------------|
| `TEA(path, SR, isRegular, ...)` | Constructor — binds to file |
| `write(t, Samples)` | Create or append time-series data |
| `write_channels(Samples, ch_map)` | Append new channels |
| `read(channels, t_range, s_range)` | Read with optional ranges and channel selection |
| `refresh()` | Recompute dependent variables |
| `info()` | Return metadata summary |
| `stream(rate)` | Read-only view of one sample rate (`read`, `N`, `C`, `channel_names`, `units`) |

| Property | Description |
|----------|-------------|
| `file_path` | Bound file path (immutable) |
| `SR` | Sample rate (immutable) |
| `isRegular` | Regularity flag (immutable) |
| `N` | Total samples (read from file) |
| `C` | Total channels (read from file) |
| `ch_map` | Channel map (read from file) |
| `rates` | Sample rates of the streams in the file |
| `schema_version` | 1 or 2 |

## Changelog

- 0.2.0 (2026-10-11): TEA schema version 2, several sample rates in one file (`tea.write`,
  `tea.Writer`, `tea.read`; MATLAB `TEA.rates`, `TEA.stream`, `TEAStream`). Version-1 files
  read as before; one-rate version-2 files read with version-1 readers.

## License

MIT
