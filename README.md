# TEA: Time-Encoded Array

A standardized timeseries data format built on HDF5/`.mat` v7.3 files. Designed for multichannel sampled data with explicit per-sample timestamps, natural discontinuity handling, and efficient partial reads.

## Quick Start (MATLAB)

```matlab
% Add to path
addpath('matlab');

% --- Write ---
SR = 1000;
t = (0:4999)' / SR;
Samples = randn(5000, 3);
brew_TEA('data.mat', t, Samples, SR, true, 't_units', 's', 'ch_map', [1 2 3]);

% --- Read ---
[Data, t_out, disc_info] = drink_TEA('data.mat', [1 3], [1.0, 2.0], []);
% disc_info.is_discontinuous, disc_info.cont, disc_info.disc

% --- Append in time ---
t2 = t(end) + (1:5000)' / SR;
brew_TEA('data.mat', t2, randn(5000,3), SR, true, 'mode', 'append_time');

% --- Append channels ---
brew_TEA('data.mat', [], randn(10000,2), SR, true, 'mode', 'append_channels');

% --- Repair missing metadata ---
refresh_TEA('data.mat');
```

## Schema

Each TEA file contains **one time axis** (`t`) and one primary data matrix (`Samples`). See [schema/tea_schema.md](schema/tea_schema.md) for the full specification.

### Required

| Variable | Shape | Description |
|----------|-------|-------------|
| `t` | `[N×1]` | Timestamps (monotonically increasing) |
| `Samples` | `[N×C]` | Data matrix (any numeric type) |
| `SR` | scalar/`[]` | Sample rate in Hz (can be empty if irregular) |
| `isRegular` | scalar | Regular sampling flag |

### Dependent (auto-computed)

`t_coarse`, `df_t_coarse`, `isContinuous`, `cont`, `disc` — computed by `brew_TEA` and `refresh_TEA`.

### Optional

`t_units`, `ch_map`, `SR_original`, `hdr`, `tea_version`, plus any user-defined fields.

## Functions

| Function | Description |
|----------|-------------|
| `brew_TEA` | Create or append to a TEA file |
| `drink_TEA` | Read data with optional time/sample range and channel selection |
| `refresh_TEA` | Compute/add missing dependent variables |

## Discontinuity Model

For regularly sampled data (`isRegular = true`), a discontinuity is any `diff(t) > 1.1/SR`. When gaps exist:
- `cont`: `[n_cont × 2]` matrix of `[start_idx, stop_idx]` for each continuous block
- `disc`: `[n_disc × 2]` matrix of `[last_before_gap, first_after_gap]` for each gap

The reader (`drink_TEA`) also detects discontinuities in the returned segment and provides local `cont`/`disc` arrays via the `disc_info` output.

## License

MIT
