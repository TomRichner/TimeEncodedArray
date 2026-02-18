# TEA: Time-Encoded Array — Schema Specification v1.0

## Overview

TEA is a timeseries data format stored in HDF5/`.mat` v7.3 files. It is designed for multichannel sampled data with explicit per-sample timestamps.

**One time axis per file.** Each TEA file contains a single `t` vector and a single `Samples` array. Additional arrays sharing the same `t` vector (same length, same time base) may also be stored. If data requires a different time axis or sample rate, it belongs in a separate TEA file.

---

## Variable Tiers

### 1. Required Variables

These must exist for a file to be a valid TEA file.

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `t` | numeric | `[N × 1]` | Timestamp per sample. Must be monotonically increasing. Units defined by `t_units`; unitless if `t_units` is absent |
| `Samples` | any numeric | `[N × C]` | Data matrix — N time samples, C channels |
| `SR` | double | scalar or `[]` | Effective sample rate in Hz. **Can be empty when `isRegular = false`** (irregular sampling has no fixed rate) |
| `isRegular` | logical | scalar | `true` if samples are regularly spaced (constant `dt` between consecutive samples within continuous segments) |

### 2. Dependent Variables

Derivable from required variables. They accelerate read performance and provide discontinuity metadata. If missing, the reader still works but may be slower. They can be computed/added by `refresh_TEA`.

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `t_coarse` | double | `[M × 1]` | Decimated timestamp vector (~1 sample/sec) for fast time-based indexing. Built as `t(1:df_t_coarse:end)` |
| `df_t_coarse` | double | scalar | Decimation factor used to build `t_coarse` from `t`. Typically `round(SR)` |
| `isContinuous` | logical | scalar | `true` if no time gaps exist. Only meaningful when `isRegular = true`. For irregular data, always `true` by convention |
| `cont` | double | `[n_cont × 2]` | Continuous block boundaries: each row is `[start_idx, stop_idx]` into `t`. Only present when `isContinuous == false` |
| `disc` | double | `[n_disc × 2]` | Discontinuity boundaries: each row is `[last_idx_before_gap, first_idx_after_gap]` into `t`. Only present when `isContinuous == false` |

### 3. Optional Variables

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `t_units` | char | string | Units of `t` (e.g., `'us'`, `'s'`, `'ms'`). If absent, `t` is unitless |
| `ch_map` | double | `[1 × C]` | Channel identity vector mapping columns of `Samples` to source channel numbers. Defaults to `1:C` if absent |
| `SR_original` | double | scalar | Original sample rate before decimation (equals `SR` if undecimated) |
| `hdr` | struct | free-form | Source-specific metadata |
| `tea_version` | char | string | Format version (e.g., `'1.0'`) |

Any additional user-defined variables may be stored alongside these.

---

## Discontinuity Model

Applies only when `isRegular = true`.

A **discontinuity** is defined as any sample pair where:

```
diff(t(k)) > 1.1 / SR
```

where `1/SR` is the expected time step (in the same units as `t`).

When discontinuities are present (`isContinuous = false`):

- **`cont`**: An `[n_cont × 2]` matrix. Each row `[start_idx, stop_idx]` defines a contiguous block of samples. The union of all blocks covers all N samples.
- **`disc`**: An `[n_disc × 2]` matrix. Each row `[last_idx_before_gap, first_idx_after_gap]` defines a gap boundary. `n_disc = n_cont - 1`.

When `isContinuous = true`, `cont` and `disc` are not stored (or `cont = [1, N]` and `disc` is empty).

When `isRegular = false`, discontinuity analysis does not apply. Irregular timestamps are inherent to the data.

---

## Flag Summary

| `isRegular` | `isContinuous` | Meaning | `cont`/`disc` |
|:-----------:|:--------------:|---------|:--------------:|
| `true` | `true` | Uniform sampling, no gaps | Not stored |
| `true` | `false` | Uniform within segments, gaps present | Stored |
| `false` | `true` (convention) | Irregular sampling | Not applicable |

---

## File Format

TEA files are HDF5 files saved with MATLAB's `-v7.3` flag (or equivalent HDF5 writers in other languages). Large arrays (`t`, `Samples`) should use chunked storage for efficient partial reads. The recommended chunk size is `[32000, 1]` for column-oriented access.
