classdef TEA < handle
% TEA Time-Encoded Array — a class for reading/writing timeseries HDF5/.mat files.
%
%   A TEA object is bound to one file and holds persistent properties (SR,
%   isRegular) so write/read calls don't repeat them.
%
%   Construction:
%       tea = TEA(file_path, SR, isRegular)
%       tea = TEA(file_path, SR, isRegular, 't_units', 's', 'hdr', hdr_struct)
%
%       If the file already exists, the constructor opens it and validates
%       that SR/isRegular match. If it doesn't exist, properties are stored
%       and the file is created on the first write.
%
%   Methods:
%       write(t, Samples)           — Append time-series data
%       write_channels(Samples, ch_map) — Append new channels
%       [D,t,disc] = read(channels, t_range, s_range) — Read data
%       refresh()                   — Recompute dependent variables
%       s = info()                  — Return file metadata summary
%
%   Example:
%       tea = TEA('data.mat', 1000, true, 't_units', 's');
%       tea.write(t1, s1);       % creates file
%       tea.write(t2, s2);       % appends
%       [D, t] = tea.read([1 3], [1.0, 2.0], []);
%
%   Single-file implementation — no external dependencies required.

    properties (SetAccess = immutable)
        file_path   char        % Path to the HDF5/.mat file
        SR          double      % Sample rate (Hz), [] for irregular
        isRegular   logical     % true = regularly spaced samples
    end

    properties
        t_units     char = ''   % Units of t ('s','us','ms', etc.)
        hdr         struct      % Free-form metadata header
        tea_version char = '1.0'
    end

    properties (SetAccess = private)
        is_initialized logical = false  % true after file has datasets
    end

    properties (Dependent)
        N           % Total sample count (read from file)
        C           % Channel count (read from file)
        ch_map      % Channel identity vector
    end

    methods

        %% ============ CONSTRUCTOR ============
        function obj = TEA(file_path, SR, isRegular, varargin)
            % Parse optional name-value pairs
            p = inputParser;
            addRequired(p, 'file_path', @ischar);
            addRequired(p, 'SR');
            addRequired(p, 'isRegular', @(x) islogical(x) && isscalar(x));
            addParameter(p, 't_units', '', @ischar);
            addParameter(p, 'hdr', struct(), @isstruct);
            addParameter(p, 'tea_version', '1.0', @ischar);
            parse(p, file_path, SR, isRegular, varargin{:});

            % Validate SR
            if isRegular
                if isempty(SR) || ~isscalar(SR) || SR <= 0
                    error('TEA:InvalidSR', 'SR must be a positive scalar when isRegular=true.');
                end
            else
                if ~isempty(SR) && (~isscalar(SR) || SR <= 0)
                    error('TEA:InvalidSR', 'SR must be a positive scalar or empty.');
                end
            end

            obj.file_path = file_path;
            obj.SR = SR;
            obj.isRegular = isRegular;
            obj.t_units = p.Results.t_units;
            obj.hdr = p.Results.hdr;
            obj.tea_version = p.Results.tea_version;

            % If file exists, validate and sync
            if exist(file_path, 'file') == 2
                mf = matfile(file_path);
                vars = {whos(mf).name};

                if ismember('SR', vars)
                    file_SR = mf.SR;
                    if ~isempty(file_SR) && ~isempty(SR) && abs(file_SR - SR) > 1e-10
                        error('TEA:SRMismatch', 'SR mismatch: file has %g but constructor says %g.', file_SR, SR);
                    end
                end
                if ismember('isRegular', vars)
                    file_isRegular = mf.isRegular;
                    if file_isRegular ~= isRegular
                        error('TEA:FlagMismatch', 'isRegular mismatch: file has %d but constructor says %d.', file_isRegular, isRegular);
                    end
                end

                % Check if datasets exist
                obj.is_initialized = ismember('t', vars) && ismember('Samples', vars);
            end
        end

        %% ============ DEPENDENT PROPERTY GETTERS ============
        function val = get.N(obj)
            if ~exist(obj.file_path, 'file') || ~obj.is_initialized
                val = 0;
                return;
            end
            mf = matfile(obj.file_path);
            info_t = whos(mf, 't');
            val = info_t.size(1);
        end

        function val = get.C(obj)
            if ~exist(obj.file_path, 'file') || ~obj.is_initialized
                val = 0;
                return;
            end
            mf = matfile(obj.file_path);
            info_s = whos(mf, 'Samples');
            val = info_s.size(2);
        end

        function val = get.ch_map(obj)
            if ~exist(obj.file_path, 'file') || ~obj.is_initialized
                val = [];
                return;
            end
            mf = matfile(obj.file_path);
            vars = {whos(mf).name};
            if ismember('ch_map', vars)
                val = mf.ch_map;
            else
                val = 1:obj.C;
            end
        end

        %% ============ INIT ============
        function init(obj, n_samples, n_channels, data_class)
        % INIT Pre-allocate the HDF5 file with chunked datasets.
        %
        %   tea.init(n_samples, n_channels, data_class)
        %
        %   Called automatically on first write if not called explicitly.

            if nargin < 4, data_class = 'double'; end

            TEA.h5_init_dataset(obj.file_path, 't', [n_samples, 1], 'double');
            TEA.h5_init_dataset(obj.file_path, 'Samples', [n_samples, n_channels], data_class);

            obj.is_initialized = true;
        end

        %% ============ WRITE ============
        function write(obj, t, Samples)
        % WRITE Append time-series data to the TEA file.
        %
        %   tea.write(t, Samples)
        %
        %   On first write, creates the file and writes metadata.
        %   On subsequent writes, appends in time.

            % --- Validate inputs ---
            obj.validate_t(t);
            N_new = length(t);
            if size(Samples, 1) ~= N_new
                error('TEA:SizeMismatch', 'Samples must have %d rows to match t, but has %d.', N_new, size(Samples, 1));
            end
            C_new = size(Samples, 2);

            if obj.isRegular
                obj.validate_regularity(t);
            end

            if ~obj.is_initialized
                % --- First write: create file ---
                obj.init(N_new, C_new, class(Samples));

                % Write data via h5write (preserves chunked storage)
                h5write(obj.file_path, '/t', double(t(:)));
                h5write(obj.file_path, '/Samples', Samples);

                % Write metadata via matfile
                mf = matfile(obj.file_path, 'Writable', true);
                mf.SR = obj.SR;
                mf.isRegular = obj.isRegular;

                % Compute and write dependents
                obj.write_dependents(mf, double(t(:)));

                % Write optionals
                if ~isempty(obj.t_units)
                    mf.t_units = obj.t_units;
                end
                if ~isempty(fieldnames(obj.hdr))
                    mf.hdr = obj.hdr;
                end
                mf.tea_version = obj.tea_version;

                fprintf('TEA file created: %s (%d samples, %d channels)\n', obj.file_path, N_new, C_new);

            else
                % --- Subsequent write: append in time ---
                mf = matfile(obj.file_path, 'Writable', true);
                info_t = whos(mf, 't');
                N_old = info_t.size(1);
                info_s = whos(mf, 'Samples');
                C_old = info_s.size(2);

                if C_new ~= C_old
                    error('TEA:ChannelMismatch', ...
                        'Channel count mismatch: file has %d but write has %d. Use write_channels to add columns.', C_old, C_new);
                end

                % Validate monotonicity across junction
                t_last = mf.t(N_old, 1);
                if t(1) <= t_last
                    error('TEA:MonotonicityViolation', ...
                        'New t must start after existing t. Last: %g, first new: %g.', t_last, t(1));
                end

                N_total = N_old + N_new;

                % Resize and append via HDF5
                obj.h5_resize_and_append(obj.file_path, '/t', [N_total, 1], double(t(:)), N_old);
                obj.h5_resize_and_append(obj.file_path, '/Samples', [N_total, C_old], Samples, N_old);

                % Incrementally update dependents (no full t reload)
                mf = matfile(obj.file_path, 'Writable', true);
                obj.update_dependents_incremental(mf, double(t(:)), t_last, N_old, N_total);

                fprintf('TEA file appended: %s (now %d samples)\n', obj.file_path, N_total);
            end
        end

        %% ============ WRITE_CHANNELS ============
        function write_channels(obj, Samples, new_ch_map)
        % WRITE_CHANNELS Append new channels (columns) to existing data.
        %
        %   tea.write_channels(Samples, ch_map)

            if ~obj.is_initialized
                error('TEA:NotInitialized', 'Cannot append channels before any data has been written.');
            end

            if nargin < 3, new_ch_map = []; end

            mf = matfile(obj.file_path, 'Writable', true);
            info_s = whos(mf, 'Samples');
            N_old = info_s.size(1);
            C_old = info_s.size(2);
            N_new = size(Samples, 1);
            C_new = size(Samples, 2);

            if N_new ~= N_old
                error('TEA:SizeMismatch', 'New channels must have %d rows, but has %d.', N_old, N_new);
            end

            C_total = C_old + C_new;

            % Resize and append columns
            obj.h5_resize_and_append_cols(obj.file_path, '/Samples', [N_old, C_total], Samples, C_old);

            % Update ch_map
            mf = matfile(obj.file_path, 'Writable', true);
            vars = {whos(mf).name};
            if ismember('ch_map', vars)
                old_map = mf.ch_map;
            else
                old_map = 1:C_old;
            end

            if ~isempty(new_ch_map)
                mf.ch_map = [old_map(:)', new_ch_map(:)'];
            else
                mf.ch_map = [old_map(:)', (C_old+1):(C_old+C_new)];
            end

            fprintf('TEA channels appended: %s (now %d channels)\n', obj.file_path, C_total);
        end

        %% ============ READ ============
        function [Data, t_out, disc_info] = read(obj, channels, t_range, s_range, data_var_name)
        % READ Read data from the TEA file.
        %
        %   [Data, t_out] = tea.read(channels, t_range, s_range)
        %   [Data, t_out, disc_info] = tea.read(...)
        %
        %   Inputs:
        %       channels  - Channel numbers (uses ch_map). [] for all.
        %       t_range   - [start, end] in t units. [] to skip.
        %       s_range   - [start, end] sample indices. [] to skip.
        %       data_var_name - (optional) Variable name, default 'Samples'

            if nargin < 2, channels = []; end
            if nargin < 3, t_range = []; end
            if nargin < 4, s_range = []; end
            if nargin < 5 || isempty(data_var_name), data_var_name = 'Samples'; end
            is_loading_samples = strcmp(data_var_name, 'Samples');

            if ~isempty(t_range) && ~isempty(s_range)
                error('TEA:RangeConflict', 'Specify either t_range or s_range, not both.');
            end

            mf = matfile(obj.file_path);
            file_vars = whos(mf);
            var_names = {file_vars.name};

            if ~ismember(data_var_name, var_names)
                error('TEA:VarNotFound', 'Variable "%s" not found.', data_var_name);
            end

            % --- Channel selection ---
            channel_indices = ':';
            if is_loading_samples && ~isempty(channels)
                if ismember('ch_map', var_names)
                    fileChMap = mf.ch_map;
                else
                    info_s = whos(mf, 'Samples');
                    fileChMap = 1:info_s.size(2);
                end
                [found, pos] = ismember(channels, fileChMap);
                if any(~found)
                    error('TEA:ChannelNotFound', 'Channels not found: %s', mat2str(channels(~found)));
                end
                channel_indices = pos;
            end

            % --- Get total samples ---
            info = whos(mf, data_var_name);
            total_samples = info.size(1);

            % --- Load ---
            if isempty(t_range) && isempty(s_range)
                if total_samples > 0
                    Data = mf.(data_var_name)(:, channel_indices);
                    t_out = mf.t(:, 1);
                else
                    Data = obj.make_empty(info, channel_indices, is_loading_samples);
                    t_out = zeros(0, 1);
                end

            elseif ~isempty(s_range)
                s_start = s_range(1);
                s_end = s_range(2);
                if s_start < 1 || s_end > total_samples || s_start > s_end
                    error('TEA:InvalidRange', 'Invalid s_range [%d, %d]. Valid: [1, %d].', s_start, s_end, total_samples);
                end
                Data = mf.(data_var_name)(s_start:s_end, channel_indices);
                t_out = mf.t(s_start:s_end, 1);

            elseif ~isempty(t_range)
                [Data, t_out] = obj.load_by_time_range(mf, t_range, channel_indices, data_var_name, total_samples, info, is_loading_samples, file_vars);
            end

            % --- Discontinuity info ---
            if nargout >= 3
                disc_info = obj.compute_disc_info(mf, t_out, file_vars);
            end
        end

        %% ============ REFRESH ============
        function refresh(obj)
        % REFRESH Recompute all dependent variables in the file.

            if ~exist(obj.file_path, 'file')
                error('TEA:FileNotFound', 'File not found: %s', obj.file_path);
            end

            mf = matfile(obj.file_path, 'Writable', true);
            t = mf.t(:, 1);
            obj.write_dependents(mf, t);

            vars = {whos(mf).name};
            if ~ismember('tea_version', vars)
                mf.tea_version = '1.0';
            end

            fprintf('TEA refreshed: %s\n', obj.file_path);
        end

        %% ============ INFO ============
        function s = info(obj)
        % INFO Return a summary struct of the TEA file.

            s = struct();
            s.file_path = obj.file_path;
            s.SR = obj.SR;
            s.isRegular = obj.isRegular;
            s.N = obj.N;
            s.C = obj.C;
            s.ch_map = obj.ch_map;
            s.t_units = obj.t_units;

            if exist(obj.file_path, 'file') == 2
                mf = matfile(obj.file_path);
                vars = {whos(mf).name};
                if ismember('isContinuous', vars)
                    s.isContinuous = mf.isContinuous;
                end
                if s.N > 0
                    s.t_first = mf.t(1, 1);
                    s.t_last = mf.t(s.N, 1);
                end
            end
        end

    end % public methods


    %% ============ PRIVATE METHODS ============
    methods (Access = private)

        function validate_t(~, t)
            if isempty(t)
                error('TEA:EmptyTimestamp', 't must not be empty.');
            end
            if ~isnumeric(t)
                error('TEA:InvalidTimestamp', 't must be numeric.');
            end
            if any(diff(t(:)) <= 0)
                error('TEA:MonotonicityViolation', 't must be strictly monotonically increasing.');
            end
        end

        function validate_regularity(obj, t)
            t = double(t(:));
            if length(t) < 2, return; end
            dt = diff(t);
            expected_dt = 1 / obj.SR;
            gap_threshold = 1.1 * expected_dt;
            regular_mask = dt <= gap_threshold;
            regular_fraction = sum(regular_mask) / length(dt);

            if regular_fraction < 0.5
                error('TEA:IrregularData', ...
                    'isRegular=true but only %.0f%% of dt values are within 1.1/SR.', regular_fraction * 100);
            end

            if any(regular_mask)
                regular_dt = dt(regular_mask);
                max_dev = max(abs(regular_dt - expected_dt)) / expected_dt;
                if max_dev > 0.1
                    error('TEA:IrregularData', ...
                        'isRegular=true but within-segment dt deviates from 1/SR by %.1f%%.', max_dev * 100);
                end
            end
        end

        function write_dependents(obj, mf, t)
            N = length(t);

            % t_coarse
            if obj.isRegular && ~isempty(obj.SR) && obj.SR > 0
                df = round(obj.SR);
            else
                df = max(1, round(N / max(1, t(end) - t(1))));
            end
            df = max(1, df);

            if N > 0
                t_coarse = t(1:df:end);
            else
                t_coarse = [];
            end
            mf.t_coarse = t_coarse;
            mf.df_t_coarse = df;

            % isContinuous, cont, disc
            if obj.isRegular && ~isempty(obj.SR) && obj.SR > 0 && N > 1
                dt = diff(t);
                gap_threshold = 1.1 / obj.SR;
                gap_mask = dt > gap_threshold;

                if any(gap_mask)
                    isCont = false;
                    gap_idx = find(gap_mask);
                    mf.disc = [gap_idx, gap_idx + 1];
                    mf.cont = [[1; gap_idx + 1], [gap_idx; N]];
                else
                    isCont = true;
                end
            else
                isCont = true;
            end
            mf.isContinuous = isCont;
        end

        function update_dependents_incremental(obj, mf, t_new, t_last, N_old, N_total)
        % UPDATE_DEPENDENTS_INCREMENTAL Update dependent vars without full t reload.
        %
        %   Uses only the new chunk (t_new), the last existing timestamp (t_last),
        %   and existing metadata to incrementally update cont/disc/isContinuous
        %   and t_coarse. Falls back to write_dependents if metadata is missing.

            vars = {whos(mf).name};

            % --- Fallback: if file lacks dependent metadata, do full recompute ---
            has_meta = ismember('isContinuous', vars) && ismember('df_t_coarse', vars);
            if ~has_meta
                warning('TEA:NoMeta', ...
                    'Dependent metadata missing. Falling back to full recompute. Run refresh() to fix.');
                t_full = mf.t(:, 1);
                obj.write_dependents(mf, t_full);
                return;
            end

            N_new = length(t_new);

            % === isContinuous / cont / disc ===
            if obj.isRegular && ~isempty(obj.SR) && obj.SR > 0
                gap_threshold = 1.1 / obj.SR;

                % Load existing discontinuity state (small data)
                old_isCont = mf.isContinuous;
                if ~old_isCont && ismember('disc', vars)
                    old_disc = mf.disc;
                else
                    old_disc = zeros(0, 2);
                end

                % Check junction gap
                new_gap_entries = zeros(0, 2);
                if (t_new(1) - t_last) > gap_threshold
                    new_gap_entries = [N_old, N_old + 1];
                end

                % Check gaps within new chunk
                if N_new > 1
                    dt_new = diff(t_new);
                    gap_mask = dt_new > gap_threshold;
                    if any(gap_mask)
                        gi = find(gap_mask);
                        new_gap_entries = [new_gap_entries; ...
                            N_old + gi(:), N_old + gi(:) + 1];
                    end
                end

                % Merge
                all_disc = [old_disc; new_gap_entries];
                if isempty(all_disc)
                    mf.isContinuous = true;
                else
                    mf.isContinuous = false;
                    mf.disc = all_disc;
                    mf.cont = [[1; all_disc(:,2)], [all_disc(:,1); N_total]];
                end

            else
                % Irregular or no SR: always continuous by convention
                mf.isContinuous = true;
            end

            % === t_coarse ===
            df = mf.df_t_coarse;
            if ismember('t_coarse', vars)
                old_t_coarse = mf.t_coarse;
            else
                old_t_coarse = [];
            end

            % Find new grid points: coarse indices are 1, 1+df, 1+2*df, ...
            last_coarse_global = 1 + floor((N_old - 1) / df) * df;
            next_coarse_global = last_coarse_global + df;
            new_coarse_globals = next_coarse_global : df : N_total;

            if ~isempty(new_coarse_globals)
                new_coarse_locals = new_coarse_globals - N_old;
                new_t_coarse = t_new(new_coarse_locals);
                mf.t_coarse = [old_t_coarse(:); new_t_coarse(:)];
            end
        end

        function [Data, t_out] = load_by_time_range(obj, mf, t_range, channel_indices, data_var_name, total_samples, info, is_loading_samples, file_vars)
            var_names = {file_vars.name};

            if total_samples == 0
                Data = obj.make_empty(info, channel_indices, is_loading_samples);
                t_out = zeros(0, 1);
                return;
            end

            t_first = mf.t(1, 1);
            t_last = mf.t(total_samples, 1);
            t_start = max(t_range(1), t_first);
            t_end = min(t_range(2), t_last);

            if t_range(1) > t_last || t_range(2) < t_first
                warning('TEA:OutOfRange', 'Requested t_range outside data range.');
                Data = obj.make_empty(info, channel_indices, is_loading_samples);
                t_out = zeros(0, 1);
                return;
            end

            has_tc = ismember('t_coarse', var_names) && ismember('df_t_coarse', var_names);
            if ~has_tc
                warning('TEA:NoCoarse', 't_coarse not found. Using full t load. Run refresh() to fix.');
            end

            % Find start_idx
            if t_start <= t_first
                start_idx = 1;
            elseif has_tc
                start_idx = obj.findIndexFromTime(mf, t_start, total_samples, 'start');
            else
                full_t = mf.t(:, 1);
                start_idx = find(full_t >= t_start, 1, 'first');
            end

            % Find end_idx
            if t_end >= t_last
                end_idx = total_samples;
            elseif has_tc
                end_idx = obj.findIndexFromTime(mf, t_end, total_samples, 'end');
            else
                if ~exist('full_t', 'var'), full_t = mf.t(:, 1); end
                end_idx = find(full_t <= t_end, 1, 'last');
            end

            if isempty(start_idx) || isempty(end_idx) || start_idx > end_idx
                Data = obj.make_empty(info, channel_indices, is_loading_samples);
                t_out = zeros(0, 1);
            else
                Data = mf.(data_var_name)(start_idx:end_idx, channel_indices);
                t_out = mf.t(start_idx:end_idx, 1);
            end
        end

        function disc_info = compute_disc_info(obj, ~, t_out, file_vars)
            var_names = {file_vars.name};
            disc_info = struct();
            NN = length(t_out);

            if NN < 2
                disc_info.is_discontinuous = false;
                disc_info.cont = zeros(0, 2);
                if NN > 0, disc_info.cont = [1, NN]; end
                disc_info.disc = zeros(0, 2);
                return;
            end

            if ~obj.isRegular
                disc_info.is_discontinuous = false;
                disc_info.cont = [1, NN];
                disc_info.disc = zeros(0, 2);
                return;
            end

            SR_val = obj.SR;
            if isempty(SR_val) || SR_val <= 0
                disc_info.is_discontinuous = false;
                disc_info.cont = [1, NN];
                disc_info.disc = zeros(0, 2);
                return;
            end

            dt = diff(t_out);
            gap_mask = dt > 1.1 / SR_val;

            if any(gap_mask)
                disc_info.is_discontinuous = true;
                gi = find(gap_mask);
                disc_info.disc = [gi, gi + 1];
                disc_info.cont = [[1; gi + 1], [gi; NN]];
            else
                disc_info.is_discontinuous = false;
                disc_info.cont = [1, NN];
                disc_info.disc = zeros(0, 2);
            end
        end

        function idx = findIndexFromTime(~, mf, target_time, total_samples, bound)
            t_coarse = mf.t_coarse;
            df_coarse = mf.df_t_coarse;

            if isempty(t_coarse) || length(t_coarse) < 2
                if strcmp(bound, 'start')
                    tw = mf.t(1:min(total_samples, 1000), 1);
                    ri = find(tw >= target_time, 1, 'first');
                    idx = ri; if isempty(ri), idx = total_samples; end
                else
                    ws = max(1, total_samples - 999);
                    tw = mf.t(ws:total_samples, 1);
                    ri = find(tw <= target_time, 1, 'last');
                    if isempty(ri), idx = ws; else, idx = ws + ri - 1; end
                end
                return;
            end

            if strcmp(bound, 'start')
                d_idx = find(t_coarse >= target_time, 1, 'first');
                if isempty(d_idx)
                    guess = total_samples;
                    ws = max(1, (length(t_coarse)-1)*df_coarse+1);
                elseif d_idx == 1
                    guess = 1; ws = 1;
                else
                    guess = (d_idx-1)*df_coarse+1;
                    ws = max(1, guess - df_coarse);
                end
                we = min(guess + df_coarse, total_samples);
                ws = min(ws, we);
                tw = mf.t(ws:we, 1);
                ri = find(tw >= target_time, 1, 'first');
                if isempty(ri)
                    idx = min(we+1, total_samples);
                else
                    idx = ws + ri - 1;
                end

            else % 'end'
                d_idx = find(t_coarse > target_time, 1, 'first');
                if isempty(d_idx)
                    guess = total_samples;
                    ws = max(1, (length(t_coarse)-1)*df_coarse+1);
                elseif d_idx == 1
                    guess = 1; ws = 1;
                else
                    guess = (d_idx-1)*df_coarse+1;
                    ws = max(1, guess - df_coarse);
                end
                we = min(guess + df_coarse, total_samples);
                ws = min(ws, we);
                tw = mf.t(ws:we, 1);
                ri = find(tw <= target_time, 1, 'last');
                if isempty(ri)
                    idx = max(1, ws-1);
                else
                    idx = ws + ri - 1;
                end
            end

            idx = max(1, min(idx, total_samples));
        end

    end % private methods


    %% ============ STATIC METHODS ============
    methods (Static, Access = private)

        function h5_init_dataset(filename, var, sz, var_class)
        % Create a chunked HDF5 dataset with unlimited max dimensions.
        %   Adapted from savefast.m by Timothy E. Holy (2013).
            [fp, fb, ext] = fileparts(filename);
            if isempty(ext), filename = fullfile(fp, [fb '.mat']); end

            if exist(filename, 'file') == 0
                dummy = 0; %#ok<NASGU>
                save(filename, '-v7.3', 'dummy', '-nocompression');
                fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
                H5L.delete(fid, 'dummy', 'H5P_DEFAULT');
                H5F.close(fid);
            end

            varname = ['/' var];
            chunk_rows = min(32000, max(1, sz(1)));
            chunk_cols = max(1, sz(2));
            h5create(filename, varname, [Inf, Inf], ...
                'DataType', var_class, ...
                'ChunkSize', [chunk_rows, chunk_cols], ...
                'Deflate', 0);

            fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            dset_id = H5D.open(fid, varname);
            H5D.set_extent(dset_id, fliplr(sz));
            H5D.close(dset_id);
            H5F.close(fid);
        end

        function h5_resize_and_append(file_path, dataset_name, new_size, new_data, row_offset)
            fid = H5F.open(file_path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            dset_id = H5D.open(fid, dataset_name);
            H5D.set_extent(dset_id, fliplr(new_size));
            space_id = H5D.get_space(dset_id);
            nr = size(new_data, 1); nc = size(new_data, 2);
            H5S.select_hyperslab(space_id, 'H5S_SELECT_SET', fliplr([row_offset, 0]), [], fliplr([nr, nc]), []);
            mem_space = H5S.create_simple(2, fliplr([nr, nc]), []);
            H5D.write(dset_id, 'H5ML_DEFAULT', mem_space, space_id, 'H5P_DEFAULT', new_data);
            H5S.close(mem_space); H5S.close(space_id); H5D.close(dset_id); H5F.close(fid);
        end

        function h5_resize_and_append_cols(file_path, dataset_name, new_size, new_data, col_offset)
            fid = H5F.open(file_path, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            dset_id = H5D.open(fid, dataset_name);
            H5D.set_extent(dset_id, fliplr(new_size));
            space_id = H5D.get_space(dset_id);
            nr = size(new_data, 1); nc = size(new_data, 2);
            H5S.select_hyperslab(space_id, 'H5S_SELECT_SET', fliplr([0, col_offset]), [], fliplr([nr, nc]), []);
            mem_space = H5S.create_simple(2, fliplr([nr, nc]), []);
            H5D.write(dset_id, 'H5ML_DEFAULT', mem_space, space_id, 'H5P_DEFAULT', new_data);
            H5S.close(mem_space); H5S.close(space_id); H5D.close(dset_id); H5F.close(fid);
        end

        function Data = make_empty(info, channel_indices, is_loading_samples)
            if is_loading_samples && ~ischar(channel_indices)
                Data = zeros(0, length(channel_indices));
            elseif length(info.size) >= 2
                Data = zeros(0, info.size(2));
            else
                Data = zeros(0, 1);
            end
        end

    end % static private methods

end
