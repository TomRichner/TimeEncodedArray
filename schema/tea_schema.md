# TEA: Time-Encoded Array — Schema Specification v2

## Overview

TEA is a timeseries data format stored in HDF5/`.mat` v7.3 files. It is designed for multichannel sampled data with explicit per-sample timestamps.

There are two schema versions:

- **Version 1: one time axis per file.** The file contains a single `t` vector and a single `Samples` array at the top level. Additional arrays sharing the same `t` vector (same length, same time base) may also be stored. Sections "Variable Tiers" to "Flag Summary" below describe this layout.
- **Version 2: one time axis per sample rate.** The file holds one group per sample rate (a *stream*) under `/rates`; each group has the version-1 layout. See [Schema Version 2](#schema-version-2-several-sample-rates). A version-2 file with one stream also carries the version-1 fields at the top level, so version-1 readers open it unchanged.

A file without `schema_version` is version 1.

---

## Variable Tiers

### 1. Required Variables

These must exist for a file to be a valid TEA file.

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `t` | numeric | `[N × 1]` | Timestamp per sample. Must be monotonically increasing. Units defined by `t_units`; unitless if `t_units` is absent |
| `Samples` | `double`, `single`, `int16`, `int32` or `int64` | `[N × C]` | Data matrix — N time samples, C channels. The reference writers store one of these five classes (readers accept any numeric class). Integer samples are counts, physical when `scale` is present (see [Storage class and scaling](#storage-class-and-scaling)) |
| `SR` | double | scalar or `[]` | Effective sample rate in samples per `t_units`. When `t_units = 's'`, this equals Hz. When `t_units = 'us'`, this is samples/μs (e.g., 0.03 for 30 kHz). **Can be empty when `isRegular = false`** |
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
| `t_offset` | int64 or double | scalar | Reference time anchor. Absolute time = `t_offset × t_offset_scale + t`. Units specified by `t_offset_units` |
| `t_offset_units` | char | string | Units of `t_offset` (e.g., `'posix_s'`, `'posix_us'`). Required when `t_offset` is present |
| `t_offset_scale` | double | scalar | Conversion factor from `t_offset` units to `t` units. Defaults to `1.0` when `t_offset_units == t_units`. Required when they differ |
| `ch_map` | double | `[1 × C]` | Channel identity vector mapping columns of `Samples` to source channel numbers. Defaults to `1:C` if absent |
| `scale` | double | `[1 × C]` | Per-channel factor from integer counts to physical values (`physical = double(Samples) .* scale + offset`). Only with an integer `Samples` class |
| `offset` | double | `[1 × C]` | Per-channel offset added after scaling. May be absent when `scale` is present (then 0) |
| `SR_original` | double | scalar | Original sample rate before decimation (equals `SR` if undecimated) |
| `hdr` | struct | free-form | Source-specific metadata |
| `tea_version` | char | string | Format version (e.g., `'1.0'`; `'2.0'` in version-2 files) |
| `schema_version` | double | scalar | Schema version; absent means 1. Version-2 files store `2` |

Any additional user-defined variables may be stored alongside these.

---

## Storage class and scaling

`Samples` is stored as one of `double`, `single`, `int16`, `int32` or `int64` (doc version 2.1; before it the class was "any numeric", and every reference writer stored `double`). Integer classes hold **counts**, typically the ADC values of the source; `single` and `double` hold physical values.

An integer stream may carry per-stream `scale` and `offset` datasets (`[1 × C]` double, one value per column of `Samples`):

```
physical = double(Samples) .* scale + offset
```

- `offset` may be absent when `scale` is present; it is then 0. `scale` and `offset` are never present with a floating `Samples` class: writers refuse that, readers ignore the pair with a warning.
- `units` (schema 2) and `hdr.units`, where used, name the **physical** unit, i.e. the unit after scaling.
- Writers given physical (floating) values for an integer class store `round((x - offset) / scale)` and refuse values that do not fit the class; integer input is stored as the counts given.
- Readers: with `scale`, `read` returns physical values as `double` by default; a `dtype` option gives `single` or the stored counts (`'int'`). Without `scale`, `read` returns `Samples` as stored, whatever its class (a `dtype` option still casts on request; asking for counts from a floating stream is an error).
- The chunk rule (next section) uses the bytes per sample of the stored class, so integer streams have proportionally more samples per chunk.

---

## Discontinuity Model

Applies only when `isRegular = true`.

A **discontinuity** is defined as any sample pair where:

```
diff(t(k)) > 1.1 / SR
```

where `1/SR` is the expected time step (in the same units as `t`).

Within a continuous segment every step satisfies `0.9 / SR <= diff(t) <= 1.1 / SR`; a larger step is a discontinuity and a smaller one is an error (writers refuse it).

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

## Time Offset Model

`t_offset` provides an optional absolute time anchor for the recording. When present, the absolute time of each sample is:

```
t_absolute = t_offset × t_offset_scale + t
```

Where:
- `t_offset` is a scalar (commonly int64 POSIX seconds)
- `t_offset_scale` converts `t_offset` into `t` units (default `1.0` when units match)
- `t` contains relative timestamps in `t_units`

**Example:** `t_offset = 1770000000` (POSIX seconds), `t` in microseconds, `t_offset_scale = 1e6`:
```
t_absolute_us = 1770000000 × 1e6 + t
```

When `t_offset` is absent, `t` is interpreted as-is (absolute or relative at the user's discretion).

---

## Float64 Precision

`t` is stored as float64. At large magnitudes, float64 quantization noise may degrade timestamp accuracy. The granularity at a value x is:

```
ULP(x) ≈ x × 2⁻⁵²
```

Writers SHOULD warn when `ULP(t[0]) / median(diff(t)) > 0.01` (1%), indicating that float64 precision is insufficient for the sample spacing at the current `t` magnitude. Using `t_offset` to reduce `t` magnitude eliminates this issue.

---

## File Format

TEA files are HDF5 files saved with MATLAB's `-v7.3` flag (or equivalent HDF5 writers in other languages). Large arrays (`t`, `Samples`) should use chunked storage for efficient partial reads. The reference implementations use chunks spanning all channels and holding about 512 KB of samples (`[n, C]` as MATLAB sees it, with `n = clamp(floor(512 KiB / (C × bytes per sample)), 64, 8192)`; 237 samples for 276 `double` channels, 8192 for one channel), which keeps short time-window reads cheap and a chunk within HDF5's default 1 MB chunk cache. `t` is chunked as a one-channel stream (8192 samples). HDF5 stores the last, partial chunk of a dataset at full size, so an uncompressed file carries up to about 512 KB plus 64 KB of padding; compression removes it. When compressed, they use DEFLATE (gzip) level 1 without the shuffle filter. Both apply to every stream of a version-2 file.

Shapes in this document are as MATLAB sees them; HDF5 stores the transpose (e.g. `Samples` `[N × C]` is the HDF5 dataset `(C, N)`).

---

## Schema Version 2: several sample rates

A recording may hold channels at different sample rates (e.g. 256 Hz EEG and 1024 Hz DC channels, or 6400 Hz and 32 kHz streams). Version 2 stores them in one file, one **stream** per sample rate.

### Layout

```
/schema_version          2
/tea_version             '2.0'
/t_units                 's'
/t_offset                POSIX seconds (absolute time = t_offset × t_offset_scale + t)
/t_offset_units          'posix_s'
/t_offset_scale          1.0
/channel_names           [C_total × L] char   all channels, stream by stream
/channel_rates           [C_total × 1] double sample rate of each of those channels
/hdr                     struct (optional, free-form)
/rates                   struct group, one subgroup per stream
/rates/r1                struct group: first stream
    t                    [N1 × 1]  double   timestamps, strictly increasing, gaps allowed
    Samples              [N1 × C1] double | single | int16 | int32 | int64
    SR                   scalar             sample rate of this stream
    isRegular            logical
    t_coarse, df_t_coarse, isContinuous, cont, disc   (as in version 1, for this stream)
    channel_names        [C1 × L] char      one row per channel, space padded
    units                [C1 × L] char      one row per channel (e.g. 'uV'), space padded;
                                            the physical unit, after scaling
    scale, offset        [1 × C1] double    optional, integer Samples only (see Storage class and scaling)
/rates/r2                second stream, same fields
...
```

- Streams are named `r1`, `r2`, … in the order they were written; readers order them by that number. Each stream has a distinct `SR`; a stream is looked up by its rate.
- Every stream group is a complete version-1 layout of the required and dependent variables, so the version-1 rules (discontinuity model, flags, `t_coarse`) apply within each stream. Gaps differ freely between streams.
- `t_units`, `t_offset`, `t_offset_units` and `t_offset_scale` are shared by all streams and live only at the top level.
- `channel_names` and `units` are char matrices so MATLAB `load` returns them without cell references; `cellstr()` turns them into cell arrays. Rows are padded with spaces; readers strip trailing spaces and NULs.
- Groups carry `MATLAB_class = 'struct'`, so `S = load(file)` gives `S.rates.r1.Samples` etc.

### One stream: version-1 compatible

When the file has exactly one stream, the top level also has `t`, `Samples`, `SR`, `isRegular`, `t_coarse`, `df_t_coarse`, `isContinuous`, (when there are gaps) `cont`, `disc` and (for a scaled integer stream) `scale`, `offset`. These are HDF5 hard links to the datasets in `/rates/r1`, so the samples are stored once. Version-1 readers (MATLAB and Python `TEA` from before version 2) read such a file unchanged; a version-1 reader that knows `scale` applies it, an older one returns the counts.

With more than one stream there are no top-level `t`, `Samples`, `SR` or `isRegular`; a version-1 reader cannot read the file and must use a version-2 reader.

### Readers and writers

- Python: `tea.write(path, [tea.Stream(rate, channel_names, units, samples, t, dtype=..., scale=..., offset=...), ...], t_offset=...)` or `tea.Writer` (`add_stream(rate, names, units, dtype=..., scale=..., offset=...)`) for writing in blocks; `tea.read(path)` opens version 1 or 2 and gives `.rates`, `.stream(rate)` and `.channel_scales`; `stream.read(..., dtype=None | 'single' | 'double' | 'int')`.
- MATLAB: `tea = TEA(path, rate, true)` with any rate in the file; `tea.rates`, `tea.stream(rate).read(channels, t_range, s_range, 'DataType', ...)`. A version-1 file is one stream (its group is `/`).
- The version-1 write methods (`TEA.write`, `write_channels`, `refresh`) refuse version-2 files.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1 | | One time axis per file. |
| 2 | 2026-10-11 | Several sample rates per file, one group per rate under `/rates`; top-level `schema_version`, `channel_names`, `channel_rates`. One-stream files keep the version-1 fields. |
| 2.1 (doc) | 2026-10-11 | Storage classes named (`double`, `single`, `int16`, `int32`, `int64`); optional per-stream `scale` and `offset` for integer counts (`physical = double(Samples) .* scale + offset`), linked at the top level of a one-stream file like the other version-1 fields; regularity rule stated (`0.9/SR <= dt <= 1.1/SR` within a segment). `schema_version` stays 2. |
