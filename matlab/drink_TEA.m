function [Data, t_out, disc_info] = drink_TEA(file_path, channels, t_range, s_range, data_var_name)
% DRINK_TEA Read data from a TEA (Time-Encoded Array) HDF5/.mat file.
%
%   [Data, t_out] = drink_TEA(file_path, channels, t_range, s_range)
%   [Data, t_out, disc_info] = drink_TEA(...)
%
%   Reads a specific portion of a TEA file using partial I/O (matfile).
%
%   Inputs:
%       file_path     - Path to the TEA .mat file
%       channels      - Vector of channel numbers (e.g., [1,3,5]). Uses
%                        ch_map if present, otherwise treats as column indices.
%                        Pass [] to load all channels.
%       t_range       - [1x2] [start_time, end_time] in t units.
%                        If provided, s_range must be empty.
%       s_range       - [1x2] [start_sample, end_sample] (1-indexed).
%                        If provided, t_range must be empty.
%       data_var_name - (Optional) Name of the data variable to load.
%                        Defaults to 'Samples'. Can load other variables
%                        that share the same time axis (e.g., 'nsd_SequenceIDs').
%
%   If both t_range and s_range are empty, the entire dataset is loaded.
%
%   Outputs:
%       Data      - Numeric matrix of the selected data.
%                    Size is [n_samples x length(channels)] or [n_samples x n_columns].
%       t_out     - Corresponding timestamp vector for the loaded samples.
%       disc_info - (Optional, computed only if requested) Struct with:
%                    .is_discontinuous : logical flag
%                    .cont             : [n_cont x 2] local indices of continuous blocks in t_out
%                    .disc             : [n_disc x 2] local indices of discontinuities in t_out
%                    Discontinuity detection uses diff(t_out) > 1.1/SR threshold.
%                    For irregular data (isRegular=false), disc_info.is_discontinuous = false.
%
%   Note: This function uses MATLAB's matfile() for efficient partial I/O.
%   If t_coarse/df_t_coarse are present, time-range lookups are accelerated.
%   If they are missing, a full t-vector load is used (with a warning).
%
%   See also: brew_TEA, refresh_TEA

    % --- Default data variable name ---
    if nargin < 5 || isempty(data_var_name)
        data_var_name = 'Samples';
    end
    is_loading_samples = strcmp(data_var_name, 'Samples');

    % --- Validate range inputs ---
    if nargin < 3, t_range = []; end
    if nargin < 4, s_range = []; end
    if ~isempty(t_range) && ~isempty(s_range)
        error('TEA:RangeConflict', 'Specify either t_range or s_range, not both.');
    end

    % --- Open file ---
    mf = matfile(file_path);
    file_vars = whos(mf);
    var_names = {file_vars.name};

    % Validate required variables
    if ~ismember(data_var_name, var_names)
        error('TEA:VarNotFound', 'Variable "%s" not found in %s.', data_var_name, file_path);
    end
    if ~ismember('t', var_names)
        error('TEA:VarNotFound', 'Timestamp variable "t" not found in %s.', file_path);
    end

    % --- Channel selection (only for Samples) ---
    if nargin < 2, channels = []; end
    channel_indices = ':';
    if is_loading_samples && ~isempty(channels)
        if ismember('ch_map', var_names)
            fileChMap = mf.ch_map;
        else
            % Default ch_map is 1:C
            info_s = whos(mf, 'Samples');
            fileChMap = 1:info_s.size(2);
        end
        [found, pos] = ismember(channels, fileChMap);
        if any(~found)
            error('TEA:ChannelNotFound', 'Channels not found: %s. Available: %s.', ...
                mat2str(channels(~found)), mat2str(fileChMap));
        end
        channel_indices = pos;
    elseif ~is_loading_samples && ~isempty(channels)
        warning('TEA:ChannelIgnored', 'Channel argument is ignored when loading "%s". Loading all columns.', data_var_name);
    end

    % --- Get total samples ---
    info = whos(mf, data_var_name);
    total_samples = info.size(1);

    % --- Load data based on range ---
    if isempty(t_range) && isempty(s_range)
        % Load everything
        if total_samples > 0
            Data = mf.(data_var_name)(:, channel_indices);
            t_out = mf.t(:, 1);
        else
            Data = make_empty(info, channel_indices, is_loading_samples);
            t_out = zeros(0, 1);
        end

    elseif ~isempty(s_range)
        % Sample range
        [Data, t_out] = load_by_sample_range(mf, s_range, channel_indices, data_var_name, total_samples, info, is_loading_samples);

    elseif ~isempty(t_range)
        % Time range
        [Data, t_out] = load_by_time_range(mf, t_range, channel_indices, data_var_name, total_samples, info, is_loading_samples, file_vars);
    end

    % --- Compute disc_info if requested ---
    if nargout >= 3
        disc_info = compute_disc_info(mf, t_out, file_vars);
    end

end


%% ========== LOAD BY SAMPLE RANGE ==========
function [Data, t_out] = load_by_sample_range(mf, s_range, channel_indices, data_var_name, total_samples, info, is_loading_samples)
    s_start = s_range(1);
    s_end = s_range(2);

    if s_start < 1 || s_end > total_samples || s_start > s_end
        error('TEA:InvalidRange', 'Invalid s_range [%d, %d]. Valid range is [1, %d].', s_start, s_end, total_samples);
    end

    Data = mf.(data_var_name)(s_start:s_end, channel_indices);
    t_out = mf.t(s_start:s_end, 1);
end


%% ========== LOAD BY TIME RANGE ==========
function [Data, t_out] = load_by_time_range(mf, t_range, channel_indices, data_var_name, total_samples, info, is_loading_samples, file_vars)
    var_names = {file_vars.name};

    if total_samples == 0
        warning('TEA:EmptyFile', 'File contains no samples. Returning empty arrays.');
        Data = make_empty(info, channel_indices, is_loading_samples);
        t_out = zeros(0, 1);
        return;
    end

    t_first = mf.t(1, 1);
    t_last = mf.t(total_samples, 1);
    t_start_req = t_range(1);
    t_end_req = t_range(2);

    % Check if completely outside range
    if t_start_req > t_last || t_end_req < t_first
        warning('TEA:OutOfRange', 'Requested t_range [%g, %g] is outside data range [%g, %g].', ...
            t_start_req, t_end_req, t_first, t_last);
        Data = make_empty(info, channel_indices, is_loading_samples);
        t_out = zeros(0, 1);
        return;
    end

    % Clip to available range
    t_start_req = max(t_start_req, t_first);
    t_end_req = min(t_end_req, t_last);

    % Check for t_coarse (accelerated search)
    has_t_coarse = ismember('t_coarse', var_names) && ismember('df_t_coarse', var_names);
    if ~has_t_coarse
        warning('TEA:NoCoarse', 't_coarse not found. Using full t-vector load (may be slow). Run refresh_TEA to fix.');
    end

    % Find start_idx
    if t_start_req <= t_first
        start_idx = 1;
    elseif has_t_coarse
        start_idx = findIndexFromTime(mf, t_start_req, total_samples, 'start');
    else
        full_t = mf.t(:, 1);
        start_idx = find(full_t >= t_start_req, 1, 'first');
        clear full_t;
    end

    % Find end_idx
    if t_end_req >= t_last
        end_idx = total_samples;
    elseif has_t_coarse
        end_idx = findIndexFromTime(mf, t_end_req, total_samples, 'end');
    else
        full_t = mf.t(:, 1);
        end_idx = find(full_t <= t_end_req, 1, 'last');
        clear full_t;
    end

    if isempty(start_idx) || isempty(end_idx) || start_idx > end_idx
        warning('TEA:NoSamples', 'No samples found in t_range [%g, %g].', t_start_req, t_end_req);
        Data = make_empty(info, channel_indices, is_loading_samples);
        t_out = zeros(0, 1);
    else
        Data = mf.(data_var_name)(start_idx:end_idx, channel_indices);
        t_out = mf.t(start_idx:end_idx, 1);
    end
end


%% ========== COMPUTE DISCONTINUITY INFO ==========
function disc_info = compute_disc_info(mf, t_out, file_vars)
    var_names = {file_vars.name};
    disc_info = struct();

    N = length(t_out);
    if N < 2
        disc_info.is_discontinuous = false;
        disc_info.cont = [1, max(N, 0)];
        disc_info.disc = zeros(0, 2);
        if N == 0
            disc_info.cont = zeros(0, 2);
        end
        return;
    end

    % Check if data is regular
    if ismember('isRegular', var_names)
        isRegular = mf.isRegular;
    else
        isRegular = true; % assume regular if not specified (legacy files)
    end

    if ~isRegular
        % Irregular data: no discontinuity concept
        disc_info.is_discontinuous = false;
        disc_info.cont = [1, N];
        disc_info.disc = zeros(0, 2);
        return;
    end

    % Get SR
    if ismember('SR', var_names)
        SR = mf.SR;
    else
        % Estimate from data
        SR = 1 / median(diff(t_out));
    end

    if isempty(SR) || SR <= 0
        disc_info.is_discontinuous = false;
        disc_info.cont = [1, N];
        disc_info.disc = zeros(0, 2);
        return;
    end

    % Detect discontinuities
    dt = diff(t_out);
    gap_threshold = 1.1 / SR;
    gap_mask = dt > gap_threshold;

    if any(gap_mask)
        disc_info.is_discontinuous = true;
        gap_indices = find(gap_mask);
        disc_info.disc = [gap_indices, gap_indices + 1]; % local indices
        block_starts = [1; gap_indices + 1];
        block_stops = [gap_indices; N];
        disc_info.cont = [block_starts, block_stops];
    else
        disc_info.is_discontinuous = false;
        disc_info.cont = [1, N];
        disc_info.disc = zeros(0, 2);
    end
end


%% ========== FIND INDEX FROM TIME (t_coarse accelerated) ==========
function idx = findIndexFromTime(mf, target_time, total_samples, bound)
% findIndexFromTime Find sample index for a target time using t_coarse.
%
%   Uses the decimated t_coarse for coarse search, then loads a small
%   window of the full t vector for fine search.

    t_coarse = mf.t_coarse;
    df_coarse = mf.df_t_coarse;

    % Handle degenerate t_coarse
    if isempty(t_coarse) || length(t_coarse) < 2
        if strcmp(bound, 'start')
            t_window = mf.t(1:min(total_samples, 1000), 1);
            rel_idx = find(t_window >= target_time, 1, 'first');
            if isempty(rel_idx)
                idx = total_samples;
            else
                idx = rel_idx;
            end
        elseif strcmp(bound, 'end')
            win_start = max(1, total_samples - 999);
            t_window = mf.t(win_start:total_samples, 1);
            rel_idx = find(t_window <= target_time, 1, 'last');
            if isempty(rel_idx)
                idx = win_start;
            else
                idx = win_start + rel_idx - 1;
            end
        else
            error('TEA:InvalidBound', 'bound must be "start" or "end".');
        end
        return;
    end

    if strcmp(bound, 'start')
        d_idx = find(t_coarse >= target_time, 1, 'first');
        if isempty(d_idx)
            guess_idx = total_samples;
            win_start = max(1, (length(t_coarse) - 1) * df_coarse + 1);
        elseif d_idx == 1
            guess_idx = 1;
            win_start = 1;
        else
            guess_idx = (d_idx - 1) * df_coarse + 1;
            win_start = max(1, guess_idx - df_coarse);
        end

        win_end = min(guess_idx + df_coarse, total_samples);
        win_start = min(win_start, win_end);
        t_window = mf.t(win_start:win_end, 1);
        rel_idx = find(t_window >= target_time, 1, 'first');

        if isempty(rel_idx)
            if win_end == total_samples
                idx = total_samples;
            else
                idx = win_end + 1;
            end
        else
            idx = win_start + rel_idx - 1;
        end

    elseif strcmp(bound, 'end')
        d_idx = find(t_coarse > target_time, 1, 'first');
        if isempty(d_idx)
            guess_idx = total_samples;
            win_start = max(1, (length(t_coarse) - 1) * df_coarse + 1);
        elseif d_idx == 1
            guess_idx = 1;
            win_start = 1;
        else
            guess_idx = (d_idx - 1) * df_coarse + 1;
            win_start = max(1, guess_idx - df_coarse);
        end

        win_end = min(guess_idx + df_coarse, total_samples);
        win_start = min(win_start, win_end);
        t_window = mf.t(win_start:win_end, 1);
        rel_idx = find(t_window <= target_time, 1, 'last');

        if isempty(rel_idx)
            if win_start == 1
                idx = 1;
            else
                idx = win_start - 1;
            end
        else
            idx = win_start + rel_idx - 1;
        end
    else
        error('TEA:InvalidBound', 'bound must be "start" or "end".');
    end

    idx = max(1, min(idx, total_samples));
end


%% ========== UTILITY ==========
function Data = make_empty(info, channel_indices, is_loading_samples)
    if is_loading_samples && ~ischar(channel_indices)
        Data = zeros(0, length(channel_indices));
    elseif length(info.size) >= 2
        Data = zeros(0, info.size(2));
    else
        Data = zeros(0, 1);
    end
end
