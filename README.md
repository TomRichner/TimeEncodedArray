# TEA: Time-Encoded Array

A standardized timeseries data format built on HDF5/`.mat` v7.3 files. Designed for multichannel sampled data with explicit per-sample timestamps, natural discontinuity handling, and efficient partial reads.

## Quick Start (MATLAB)

```matlab
addpath('matlab');

% --- Create and write ---
tea = TEA('data.mat', 1000, true, 't_units', 's');
tea.write(t, Samples);                % creates file on first call

% --- Append ---
tea.write(t2, Samples2);             % appends in time (no SR/mode needed)
tea.write_channels(new_data, [5 6]); % appends channels

% --- Read ---
[Data, t_out, disc_info] = tea.read([1 3], [1.0, 2.0], []);

% --- Info ---
s = tea.info();   % .N, .C, .SR, .isContinuous, ...

% --- Re-open existing file ---
tea2 = TEA('data.mat', 1000, true);  % validates SR/isRegular match
tea2.write(t3, s3);                  % appends
```

## Schema

See [schema/tea_schema.md](schema/tea_schema.md) for the full specification.

### Required

| Variable | Shape | Description |
|----------|-------|-------------|
| `t` | `[N×1]` | Timestamps (monotonically increasing) |
| `Samples` | `[N×C]` | Data matrix (any numeric type) |
| `SR` | scalar/`[]` | Sample rate in Hz (can be empty if irregular) |
| `isRegular` | scalar | Regular sampling flag |

### Dependent (auto-computed)

`t_coarse`, `df_t_coarse`, `isContinuous`, `cont`, `disc`

### Optional

`t_units`, `ch_map`, `SR_original`, `hdr`, `tea_version`

## Class API

| Method | Description |
|--------|-------------|
| `TEA(path, SR, isRegular, ...)` | Constructor — binds to file |
| `write(t, Samples)` | Create or append time-series data |
| `write_channels(Samples, ch_map)` | Append new channels |
| `read(channels, t_range, s_range)` | Read with optional ranges and channel selection |
| `refresh()` | Recompute dependent variables |
| `info()` | Return metadata summary struct |

| Property | Description |
|----------|-------------|
| `file_path` | Bound file path (immutable) |
| `SR` | Sample rate (immutable) |
| `isRegular` | Regularity flag (immutable) |
| `N` | Total samples (read from file) |
| `C` | Total channels (read from file) |
| `ch_map` | Channel map (read from file) |

## License

MIT
