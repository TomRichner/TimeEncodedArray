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

# --- Integer counts with a per-channel scale (physical = counts * scale + offset) ---
tea = TEA('counts.mat', 1000, True, dtype=np.int32, scale=[0.1, 0.2], offset=[0, 5])
tea.write(t, x)                      # float input is rounded to counts; int input stored as given
data, t_out, _ = tea.read()          # physical float64 (scale applied)
counts, _, _ = tea.read(dtype='int') # the stored int32 counts
```

### Several sample rates in one file (schema version 2)

```python
import tea

# --- Write: one Stream per sample rate; t in seconds relative to t_offset, gaps allowed ---
tea.write('clip.mat', [
    tea.Stream(256,  ['Fp1', 'Fp2', 'Cz'], 'uV', x256, t256),    # samples (n, ch), t (n,); float64 by default
    tea.Stream(1024, ['DC1', 'DC2'], ['mV', 'uV'], x1024, t1024, dtype=np.float32),
    tea.Stream(2048, ['R1'], 'uV', counts, t2048, dtype=np.int32, scale=0.0298),  # counts + scale
], t_offset=1770000000, compress=True)

# --- Or in blocks (large recordings) ---
with tea.Writer('clip.mat', t_offset=1770000000) as w:
    w.add_stream(256, names256, 'uV')                                  # float64
    w.add_stream(1024, names1024, 'uV', dtype=np.int32, scale=gains, offset=offsets)
    for t, x in blocks:
        w.append(256, t, x)

# --- Read (version 1 or 2) ---
f = tea.read('clip.mat')
f.rates                                    # [256.0, 1024.0, 2048.0]
s = f.stream(1024)                         # s.channel_names, s.units, s.N, s.C, s.dtype, s.scale, s.offset
data, t, disc = s.read(channels=['DC2'], t_range=(10.0, 12.0))
data, t, disc = s.read(dtype=np.float32)   # physical single; dtype='int' gives the stored counts
f.channel_scales                           # scale per channel, None for unscaled streams

TEA('clip.mat', 256, True).stream(1024)    # same, from the version-1 class
```

Storage classes: `dtype` is `float64` (default), `float32`, `int16`, `int32` or `int64`. `scale` and
`offset` (one value or one per channel) go with an integer dtype only; `units` is the physical unit.
`read(dtype=None)` returns physical float64 when the stream has `scale`, otherwise the samples as
stored; `np.float32`/`np.float64` give physical values in that type; `'int'` gives the counts.

A file written with one stream also carries the version-1 fields (including `scale`/`offset`), so
`TEA(path, SR, True).read()` (and older readers) work on it unchanged. A version-1 file reads through
`tea.read` as one stream.

### Running Tests

```bash
uv run pytest -q                                 # 65 tests: test_tea.py (41) and test_tea_v2.py (24)
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

% --- Integer counts with a per-channel scale (physical = double(counts) .* scale + offset) ---
tea = TEA('counts.mat', 1000, true, 'scale', [0.1 0.2], 'offset', [0 5]);
tea.write(t, int32(counts));         % Samples is stored as the class given (double, single, int16/32/64)
Data = tea.read();                   % physical double (scale applied)
Data = tea.read([], [], [], 'DataType', 'int');     % the stored int32 counts ('single' also accepted)
```

### Several sample rates (schema version 2)

```matlab
tea = TEA('clip.mat', 256, true);    % any rate in the file
tea.rates                            % [256 1024]
s = tea.stream(1024);                % TEAStream: s.channel_names, s.units, s.N, s.C, s.dtype, s.scale, s.offset
[Data, t, disc_info] = s.read({'DC2'}, [10.0, 12.0], []);
Data = s.read([], [], [], 'DataType', 'single');   % physical single; 'int' gives the counts
```

Version-2 files are written from Python. `load('clip.mat')` gives `S.rates.r1.Samples`, `S.rates.r1.t`, ...

### Running Tests

```matlab
cd matlab/tests
runtests('test_tea')                 % 38 unit tests
runtests('test_tea_v2')              % 8 tests; uses files from python/tests/write_v2_test_cases.py
```

## Cross-Language Validation

Both implementations write 10 deterministic test cases (case 10: int32 counts with scale and offset).
Two comparator scripts verify every variable matches:

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
| `Samples` | `[N×C]` | Data matrix: `double`, `single`, `int16`, `int32` or `int64` (integer = counts) |
| `SR` | scalar/`[]` | Sample rate in samples per `t_units` (= Hz when `t_units='s'`) |
| `isRegular` | scalar | Regular sampling flag |

### Dependent (auto-computed)

`t_coarse`, `df_t_coarse`, `isContinuous`, `cont`, `disc`

### Optional

`t_units`, `t_offset`, `t_offset_units`, `t_offset_scale`, `ch_map`, `scale`, `offset` (`[1×C]`, integer
`Samples` only: physical = `double(Samples) .* scale + offset`), `SR_original`, `hdr`, `tea_version`

## API

| Method | Description |
|--------|-------------|
| `TEA(path, SR, isRegular, ...)` | Constructor — binds to file (`dtype` in Python; `scale`, `offset` in both) |
| `write(t, Samples)` | Create or append time-series data |
| `write_channels(Samples, ch_map)` | Append new channels |
| `read(channels, t_range, s_range, dtype / 'DataType')` | Read with optional ranges, channel selection and sample class |
| `refresh()` | Recompute dependent variables |
| `info()` | Return metadata summary (includes `dtype` and `scaled`) |
| `stream(rate)` | Read-only view of one sample rate (`read`, `N`, `C`, `channel_names`, `units`, `dtype`, `scale`, `offset`) |

| Property | Description |
|----------|-------------|
| `file_path` | Bound file path (immutable) |
| `SR` | Sample rate (immutable) |
| `isRegular` | Regularity flag (immutable) |
| `N` | Total samples (read from file) |
| `C` | Total channels (read from file) |
| `ch_map` | Channel map (read from file) |
| `dtype`, `scale`, `offset` | Storage class and per-channel scaling of an integer file |
| `rates` | Sample rates of the streams in the file |
| `schema_version` | 1 or 2 |

## Storage size

One stream, gzip level 1, whole-stream read in Python (best of 3), 276 channels × 245 760 samples
(492 s) of EEG quantised to 0.0298 µV counts:

| Samples class | uncompressed | gzip 1 | gzip 1 + shuffle |
|---------------|-------------:|-------:|-----------------:|
| float64 | 545 MB | 273 MB (read 2.3 s) | 438 MB |
| float32 | 274 MB | 197 MB (1.3 s) | 199 MB |
| int32 counts + scale | 274 MB | 138 MB (1.6 s) | **103 MB (0.9 s)** |

Compressed files therefore use the shuffle filter for integer classes only; floating classes use plain
DEFLATE (the measurement script is `myndos/scratch/tea_int32_shuffle_measure.py`).

## Changelog

- 0.3.0 (2026-10-11): storage classes `float64`, `float32`, `int16`, `int32`, `int64` with optional
  per-channel `scale`/`offset` for integer counts (Python `dtype=`/`scale=`/`offset=` on `TEA`,
  `Stream` and `Writer.add_stream`; MATLAB `'scale'`/`'offset'`); readers return physical values by
  default and counts or single on request (`dtype=` / `'DataType'`); shuffle filter for compressed
  integer data; `tea_h5.py` holds the HDF5 helpers. Schema doc 2.1 (schema version stays 2).
- 0.2.0 (2026-10-11): TEA schema version 2, several sample rates in one file (`tea.write`,
  `tea.Writer`, `tea.read`; MATLAB `TEA.rates`, `TEA.stream`, `TEAStream`). Version-1 files
  read as before; one-rate version-2 files read with version-1 readers.

## License

MIT
