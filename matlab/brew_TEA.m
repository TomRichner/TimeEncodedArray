function brew_TEA(file_path, t, Samples, SR, isRegular, varargin)
% BREW_TEA Create or append to a TEA (Time-Encoded Array) HDF5/.mat file.
%
%   brew_TEA(file_path, t, Samples, SR, isRegular)
%   brew_TEA(file_path, t, Samples, SR, isRegular, Name, Value, ...)
%
%   Creates a new TEA file or appends to an existing one.
%
%   Required Inputs:
%       file_path  - Path to the .mat file (created if new)
%       t          - [N x 1] timestamp vector (monotonically increasing)
%       Samples    - [N x C] data matrix (any numeric type)
%       SR         - Sample rate in Hz (scalar). Can be [] if isRegular=false
%       isRegular  - Logical scalar. true = regularly spaced samples
%
%   Name-Value Pairs (optional):
%       'mode'         - 'create' (default for new file), 'append_time',
%                        or 'append_channels'
%       't_units'      - String, e.g. 'us', 's', 'ms'. If omitted, t is unitless
%       'ch_map'       - [1 x C] channel identity vector
%       'SR_original'  - Original sample rate before decimation
%       'hdr'          - Struct of free-form metadata
%       'tea_version'  - Format version string (default '1.0')
%
%   Examples:
%       % Create a new file
%       brew_TEA('data.mat', t, samples, 1000, true, 't_units', 'us')
%
%       % Append more time samples
%       brew_TEA('data.mat', t_new, samples_new, 1000, true, 'mode', 'append_time')
%
%       % Append more channels
%       brew_TEA('data.mat', [], new_ch_data, 1000, true, 'mode', 'append_channels', 'ch_map', [5 6])
%
%   See also: drink_TEA, refresh_TEA

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'file_path', @ischar);
    addRequired(p, 't');
    addRequired(p, 'Samples');
    addRequired(p, 'SR');
    addRequired(p, 'isRegular', @(x) islogical(x) && isscalar(x));
    addParameter(p, 'mode', '', @ischar);
    addParameter(p, 't_units', '', @ischar);
    addParameter(p, 'ch_map', [], @isnumeric);
    addParameter(p, 'SR_original', [], @isnumeric);
    addParameter(p, 'hdr', [], @isstruct);
    addParameter(p, 'tea_version', '1.0', @ischar);
    parse(p, file_path, t, Samples, SR, isRegular, varargin{:});
    opts = p.Results;

    % Determine mode
    file_exists = exist(file_path, 'file') == 2;
    if isempty(opts.mode)
        if file_exists
            error('TEA:FileExists', ...
                'File already exists: %s. Specify mode=''append_time'' or ''append_channels'' to append, or delete the file first.', file_path);
        end
        opts.mode = 'create';
    end

    switch opts.mode
        case 'create'
            create_new(file_path, t, Samples, SR, isRegular, opts);
        case 'append_time'
            if ~file_exists
                error('TEA:FileNotFound', 'Cannot append: file does not exist: %s', file_path);
            end
            append_time(file_path, t, Samples, SR, isRegular, opts);
        case 'append_channels'
            if ~file_exists
                error('TEA:FileNotFound', 'Cannot append: file does not exist: %s', file_path);
            end
            append_channels(file_path, Samples, opts);
        otherwise
            error('TEA:InvalidMode', 'Unknown mode: %s. Use ''create'', ''append_time'', or ''append_channels''.', opts.mode);
    end

end


%% ========== CREATE NEW FILE ==========
function create_new(file_path, t, Samples, SR, isRegular, opts)

    % --- Validate ---
    validate_t(t);
    N = length(t);
    if size(Samples, 1) ~= N
        error('TEA:SizeMismatch', 'Samples must have %d rows to match t, but has %d.', N, size(Samples, 1));
    end
    C = size(Samples, 2);
    validate_SR(SR, isRegular);

    if isRegular
        validate_regularity(t, SR);
    end

    % --- Determine data class ---
    data_class = class(Samples);

    % --- Preallocate HDF5 file with chunked, resizable datasets ---
    initialize_savefast_chunk_4_column(file_path, 't', [N, 1], 'double');
    initialize_savefast_chunk_4_column(file_path, 'Samples', [N, C], data_class);

    % --- Write large arrays via h5write (preserves chunked storage) ---
    h5write(file_path, '/t', double(t(:)));
    h5write(file_path, '/Samples', Samples);

    % --- Write scalar/small required variables via matfile ---
    mf = matfile(file_path, 'Writable', true);
    mf.SR = SR;
    mf.isRegular = isRegular;

    % --- Compute and write dependent variables ---
    t_col = double(t(:));
    write_dependents(mf, t_col, SR, isRegular);

    % --- Write optional variables ---
    write_optionals(mf, opts, C);

    fprintf('TEA file created: %s (%d samples, %d channels)\n', file_path, N, C);

end


%% ========== APPEND IN TIME ==========
function append_time(file_path, t_new, Samples_new, SR, isRegular, opts)

    % --- Validate new data ---
    validate_t(t_new);
    N_new = length(t_new);
    if size(Samples_new, 1) ~= N_new
        error('TEA:SizeMismatch', 'Samples must have %d rows to match t, but has %d.', N_new, size(Samples_new, 1));
    end

    % --- Load existing file info ---
    mf = matfile(file_path, 'Writable', true);
    existing_isRegular = mf.isRegular;
    if isRegular ~= existing_isRegular
        error('TEA:FlagMismatch', 'isRegular mismatch: file has %d but input is %d.', existing_isRegular, isRegular);
    end

    info_t = whos(mf, 't');
    N_old = info_t.size(1);
    info_s = whos(mf, 'Samples');
    C_old = info_s.size(2);
    C_new = size(Samples_new, 2);
    if C_new ~= C_old
        error('TEA:ChannelMismatch', 'Channel count mismatch: file has %d but input has %d. Use append_channels instead.', C_old, C_new);
    end

    % Validate monotonicity across junction
    t_last_existing = mf.t(N_old, 1);
    if t_new(1) <= t_last_existing
        error('TEA:MonotonicityViolation', ...
            'New t must start after existing t. Last existing: %g, first new: %g.', t_last_existing, t_new(1));
    end

    if isRegular
        validate_regularity(t_new, SR);
    end

    % --- Resize and append using HDF5 low-level API ---
    t_new_col = double(t_new(:));
    N_total = N_old + N_new;

    % Resize t dataset
    h5_resize_and_append(file_path, '/t', [N_total, 1], t_new_col, N_old);

    % Resize Samples dataset
    h5_resize_and_append(file_path, '/Samples', [N_total, C_old], Samples_new, N_old);

    % --- Recompute all dependent variables from full t ---
    % For large files, we load the full t to recompute dependents
    mf = matfile(file_path, 'Writable', true); % refresh handle
    t_full = mf.t(:, 1);
    write_dependents(mf, t_full, SR, isRegular);

    fprintf('TEA file appended in time: %s (now %d samples)\n', file_path, N_total);

end


%% ========== APPEND CHANNELS ==========
function append_channels(file_path, Samples_new, opts)

    mf = matfile(file_path, 'Writable', true);
    info_s = whos(mf, 'Samples');
    N_old = info_s.size(1);
    C_old = info_s.size(2);
    N_new = size(Samples_new, 1);
    C_new = size(Samples_new, 2);

    if N_new ~= N_old
        error('TEA:SizeMismatch', 'New channels must have %d rows to match existing, but has %d.', N_old, N_new);
    end

    C_total = C_old + C_new;

    % Resize Samples dataset and append new columns
    h5_resize_and_append_cols(file_path, '/Samples', [N_old, C_total], Samples_new, C_old);

    % Update ch_map
    mf = matfile(file_path, 'Writable', true); % refresh
    file_vars = whos(mf);
    if ismember('ch_map', {file_vars.name})
        old_ch_map = mf.ch_map;
    else
        old_ch_map = 1:C_old;
    end

    if ~isempty(opts.ch_map)
        new_ch_map = [old_ch_map(:)', opts.ch_map(:)'];
    else
        new_ch_map = [old_ch_map(:)', (C_old+1):(C_old+C_new)];
    end
    mf.ch_map = new_ch_map;

    fprintf('TEA file appended channels: %s (now %d channels)\n', file_path, C_total);

end


%% ========== HELPER FUNCTIONS ==========

function validate_t(t)
    if isempty(t)
        error('TEA:EmptyTimestamp', 't must not be empty.');
    end
    if ~isnumeric(t)
        error('TEA:InvalidTimestamp', 't must be numeric.');
    end
    t = t(:);
    if any(diff(t) <= 0)
        error('TEA:MonotonicityViolation', 't must be strictly monotonically increasing.');
    end
end

function validate_SR(SR, isRegular)
    if isRegular
        if isempty(SR) || ~isscalar(SR) || SR <= 0
            error('TEA:InvalidSR', 'SR must be a positive scalar when isRegular=true.');
        end
    else
        % SR can be empty for irregular data
        if ~isempty(SR) && (~isscalar(SR) || SR <= 0)
            error('TEA:InvalidSR', 'SR must be a positive scalar or empty.');
        end
    end
end

function validate_regularity(t, SR)
    % Check that t is consistent with isRegular=true.
    % Two checks:
    %   1. At least 50% of dt values must be "regular" (within 1.1/SR)
    %   2. Those regular dt values must not deviate from 1/SR by more than 10%
    t = double(t(:));
    if length(t) < 2
        return;
    end
    dt = diff(t);
    expected_dt = 1 / SR;
    gap_threshold = 1.1 * expected_dt;
    regular_mask = dt <= gap_threshold;
    regular_fraction = sum(regular_mask) / length(dt);

    % Check 1: majority of samples should be regularly spaced
    if regular_fraction < 0.5
        error('TEA:IrregularData', ...
            'isRegular=true but only %.0f%% of dt values are within 1.1/SR. Set isRegular=false or fix data.', ...
            regular_fraction * 100);
    end

    % Check 2: regular segments should be close to expected_dt
    if any(regular_mask)
        regular_dt = dt(regular_mask);
        max_deviation = max(abs(regular_dt - expected_dt)) / expected_dt;
        if max_deviation > 0.1
            error('TEA:IrregularData', ...
                'isRegular=true but within-segment dt deviates from 1/SR by %.1f%%. Set isRegular=false or fix data.', ...
                max_deviation * 100);
        end
    end
end

function write_dependents(mf, t, SR, isRegular)
    N = length(t);

    % --- t_coarse and df_t_coarse ---
    if isRegular && ~isempty(SR) && SR > 0
        df_t_coarse = round(SR); % ~1 sample per second
    else
        % For irregular data, pick a reasonable decimation
        df_t_coarse = max(1, round(N / max(1, (t(end) - t(1))))); % rough estimate
        if df_t_coarse < 1
            df_t_coarse = 1;
        end
    end
    df_t_coarse = max(1, df_t_coarse);

    if N > 0
        t_coarse = t(1:df_t_coarse:end);
    else
        t_coarse = [];
    end
    mf.t_coarse = t_coarse;
    mf.df_t_coarse = df_t_coarse;

    % --- isContinuous, cont, disc ---
    if isRegular && ~isempty(SR) && SR > 0 && N > 1
        dt = diff(t);
        gap_threshold = 1.1 / SR;
        gap_mask = dt > gap_threshold;

        if any(gap_mask)
            isContinuous = false;
            gap_indices = find(gap_mask);
            n_disc = length(gap_indices);

            % disc: [last_before_gap, first_after_gap]
            disc_arr = [gap_indices, gap_indices + 1];

            % cont: [block_start, block_stop]
            block_starts = [1; gap_indices + 1];
            block_stops = [gap_indices; N];
            cont_arr = [block_starts, block_stops];
        else
            isContinuous = true;
            disc_arr = zeros(0, 2);
            cont_arr = [1, N];
        end
    else
        % Irregular data or single sample: continuous by convention
        isContinuous = true;
        disc_arr = zeros(0, 2);
        if N > 0
            cont_arr = [1, N];
        else
            cont_arr = zeros(0, 2);
        end
    end

    mf.isContinuous = isContinuous;

    % Only store cont/disc if discontinuous
    if ~isContinuous
        mf.cont = cont_arr;
        mf.disc = disc_arr;
    end
end

function write_optionals(mf, opts, C)
    if ~isempty(opts.t_units)
        mf.t_units = opts.t_units;
    end
    if ~isempty(opts.ch_map)
        mf.ch_map = opts.ch_map;
    end
    if ~isempty(opts.SR_original)
        mf.SR_original = opts.SR_original;
    end
    if ~isempty(opts.hdr)
        mf.hdr = opts.hdr;
    end
    mf.tea_version = opts.tea_version;
end


%% ========== HDF5 RESIZE HELPERS ==========

function h5_resize_and_append(file_path, dataset_name, new_size, new_data, row_offset)
% Resize an HDF5 dataset and write new_data starting at row_offset+1.
% Requires the dataset to have been created with chunked storage.

    % Open file and dataset
    fid = H5F.open(file_path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    dset_id = H5D.open(fid, dataset_name);

    % Extend the dataset
    H5D.set_extent(dset_id, fliplr(new_size)); % HDF5 uses row-major order

    % Select hyperslab for writing
    space_id = H5D.get_space(dset_id);
    n_rows_new = size(new_data, 1);
    n_cols = size(new_data, 2);
    start = fliplr([row_offset, 0]); % 0-indexed, row-major
    count = fliplr([n_rows_new, n_cols]);
    H5S.select_hyperslab(space_id, 'H5S_SELECT_SET', start, [], count, []);

    % Create memory dataspace
    mem_space = H5S.create_simple(2, fliplr([n_rows_new, n_cols]), []);

    % Determine HDF5 type from MATLAB class
    h5_type = matlab_to_h5_type(class(new_data));

    % Write
    H5D.write(dset_id, h5_type, mem_space, space_id, 'H5P_DEFAULT', new_data);

    % Cleanup
    H5S.close(mem_space);
    H5S.close(space_id);
    H5D.close(dset_id);
    H5F.close(fid);
end

function h5_resize_and_append_cols(file_path, dataset_name, new_size, new_data, col_offset)
% Resize an HDF5 dataset and write new_data starting at col_offset+1.

    fid = H5F.open(file_path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    dset_id = H5D.open(fid, dataset_name);

    H5D.set_extent(dset_id, fliplr(new_size));

    space_id = H5D.get_space(dset_id);
    n_rows = size(new_data, 1);
    n_cols_new = size(new_data, 2);
    start = fliplr([0, col_offset]);
    count = fliplr([n_rows, n_cols_new]);
    H5S.select_hyperslab(space_id, 'H5S_SELECT_SET', start, [], count, []);

    mem_space = H5S.create_simple(2, fliplr([n_rows, n_cols_new]), []);

    h5_type = matlab_to_h5_type(class(new_data));

    H5D.write(dset_id, h5_type, mem_space, space_id, 'H5P_DEFAULT', new_data);

    H5S.close(mem_space);
    H5S.close(space_id);
    H5D.close(dset_id);
    H5F.close(fid);
end

function h5_type = matlab_to_h5_type(matlab_class)
    switch matlab_class
        case 'double'
            h5_type = 'H5ML_DEFAULT';
        case 'single'
            h5_type = 'H5ML_DEFAULT';
        case 'int16'
            h5_type = 'H5ML_DEFAULT';
        case 'int32'
            h5_type = 'H5ML_DEFAULT';
        case 'int64'
            h5_type = 'H5ML_DEFAULT';
        case 'uint16'
            h5_type = 'H5ML_DEFAULT';
        case 'uint32'
            h5_type = 'H5ML_DEFAULT';
        case 'uint64'
            h5_type = 'H5ML_DEFAULT';
        otherwise
            h5_type = 'H5ML_DEFAULT';
    end
end
