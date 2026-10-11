classdef TEAStream < handle
% TEASTREAM Read-only view of one stream (one sample rate) of a TEA file.
%
%   Obtained from a TEA object:
%       tea = TEA('clip.mat', 256, true);   % any rate in the file
%       tea.rates                            % e.g. [256 1024]
%       s = tea.stream(1024);
%       [Data, t, disc_info] = s.read(channels, t_range, s_range);
%
%   Works for schema version 2 files (one group per rate under /rates) and for
%   version 1 files (the whole file is one stream). Reads only the requested
%   samples (h5read with start/count), using t_coarse to find time ranges.
%
%   channels: [] for all; channel numbers (1-based, or the file's ch_map for a
%   version 1 file); or channel names as a cell array or string array.
%   t_range:  [start, end] in t units (seconds relative to t_offset).
%   s_range:  [first, last] sample indices, 1-based, inclusive.

    properties (SetAccess = private)
        file_path      char = ''
        group          char = '/'    % HDF5 group holding t and Samples
        rate           double        % sample rate, [] when irregular without SR
        isRegular      logical = true
        channel_names  cell = {}     % 1 x C cell of char
        units          cell = {}     % 1 x C cell of char ('' when unknown)
        schema_version double = 1
        t_units        char = ''
        t_offset                     % absolute time = t_offset * t_offset_scale + t
        t_offset_units char = ''
        t_offset_scale double = 1.0
    end

    properties (Dependent)
        N   % sample count
        C   % channel count
    end

    methods

        function obj = TEAStream(file_path, group)
            if nargin == 0, return; end
            obj.file_path = file_path;
            obj.group = group;
            obj.schema_version = TEAStream.schema_version_of(file_path);
            obj.t_units = TEAStream.read_string(file_path, '/t_units');
            obj.t_offset = TEAStream.read_scalar(file_path, '/t_offset');
            obj.t_offset_units = TEAStream.read_string(file_path, '/t_offset_units');
            sc = TEAStream.read_scalar(file_path, '/t_offset_scale');
            if ~isempty(sc), obj.t_offset_scale = sc; end
            obj.rate = TEAStream.read_scalar(file_path, obj.p('SR'));
            ir = TEAStream.read_scalar(file_path, obj.p('isRegular'));
            if ~isempty(ir), obj.isRegular = logical(ir); end

            C = obj.C;
            if strcmp(group, '/')
                names = TEAStream.read_names(file_path, '/hdr/channel_names');
                if numel(names) ~= C
                    names = arrayfun(@(n) sprintf('Ch%d', n), obj.ch_map(), 'UniformOutput', false);
                end
                obj.channel_names = names(:)';
                obj.units = repmat({''}, 1, C);
            else
                obj.channel_names = reshape(TEAStream.read_names(file_path, obj.p('channel_names')), 1, []);
                obj.units = reshape(TEAStream.read_names(file_path, obj.p('units')), 1, []);
            end
        end

        function val = get.N(obj)
            sz = TEAStream.dims(obj.file_path, obj.p('t'));
            val = sz(1);
        end

        function val = get.C(obj)
            sz = TEAStream.dims(obj.file_path, obj.p('Samples'));
            val = sz(2);
        end

        function [Data, t_out, disc_info] = read(obj, channels, t_range, s_range)
        % READ Read samples of this stream.
        %   [Data, t_out, disc_info] = s.read(channels, t_range, s_range)
            if nargin < 2, channels = []; end
            if nargin < 3, t_range = []; end
            if nargin < 4, s_range = []; end
            if ~isempty(t_range) && ~isempty(s_range)
                error('TEA:RangeConflict', 'Specify either t_range or s_range, not both.');
            end

            C = obj.C;
            NN = obj.N;
            idx = obj.resolve_channels(channels, C);

            if ~isempty(s_range)
                s0 = s_range(1); s1 = s_range(2);
                if s0 < 1 || s1 > NN || s0 > s1
                    error('TEA:InvalidRange', 'Invalid s_range [%d, %d]. Valid: [1, %d].', s0, s1, NN);
                end
            elseif ~isempty(t_range)
                [s0, s1] = obj.find_range(t_range, NN);
            else
                s0 = 1; s1 = NN;
            end

            if NN == 0 || s1 < s0
                Data = zeros(0, numel(idx));
                t_out = zeros(0, 1);
            else
                n = s1 - s0 + 1;
                t_out = h5read(obj.file_path, obj.p('t'), [s0 1], [n 1]);
                Data = h5read(obj.file_path, obj.p('Samples'), [s0 1], [n C]);
                Data = Data(:, idx);
            end

            if nargout >= 3
                disc_info = obj.disc_info_of(t_out);
            end
        end

        function s = info(obj)
        % INFO Summary struct of this stream.
            s = struct('file_path', obj.file_path, 'group', obj.group, ...
                'rate', obj.rate, 'SR', obj.rate, 'isRegular', obj.isRegular, ...
                'N', obj.N, 'C', obj.C, 'channel_names', {obj.channel_names}, ...
                'units', {obj.units}, 'schema_version', obj.schema_version, ...
                't_units', obj.t_units, 't_offset', obj.t_offset, ...
                't_offset_units', obj.t_offset_units, 't_offset_scale', obj.t_offset_scale);
            ic = TEAStream.read_scalar(obj.file_path, obj.p('isContinuous'));
            if ~isempty(ic), s.isContinuous = logical(ic); end
            if s.N > 0
                s.t_first = h5read(obj.file_path, obj.p('t'), [1 1], [1 1]);
                s.t_last = h5read(obj.file_path, obj.p('t'), [s.N 1], [1 1]);
            end
        end

    end % public methods

    methods (Access = private)

        function path = p(obj, name)
            if strcmp(obj.group, '/')
                path = ['/' name];
            else
                path = [obj.group '/' name];
            end
        end

        function m = ch_map(obj)
            if TEAStream.exists(obj.file_path, obj.p('ch_map'))
                m = double(h5read(obj.file_path, obj.p('ch_map')));
                m = m(:)';
            else
                m = 1:obj.C;
            end
        end

        function idx = resolve_channels(obj, channels, C)
            if isempty(channels)
                idx = 1:C;
                return;
            end
            if isstring(channels), channels = cellstr(channels); end
            if ischar(channels), channels = {channels}; end
            if ~iscell(channels), channels = num2cell(channels); end
            map = [];
            idx = zeros(1, numel(channels));
            for k = 1:numel(channels)
                ch = channels{k};
                if ischar(ch) || isstring(ch)
                    pos = find(strcmp(obj.channel_names, char(ch)), 1);
                    if isempty(pos)
                        error('TEA:ChannelNotFound', 'Channel "%s" not found.', char(ch));
                    end
                else
                    if isempty(map), map = obj.ch_map(); end
                    pos = find(map == ch, 1);
                    if isempty(pos)
                        error('TEA:ChannelNotFound', 'Channel %g not found.', ch);
                    end
                end
                idx(k) = pos;
            end
        end

        function [s0, s1] = find_range(obj, t_range, NN)
            s0 = 1; s1 = 0;
            if NN == 0, return; end
            t_first = h5read(obj.file_path, obj.p('t'), [1 1], [1 1]);
            t_last = h5read(obj.file_path, obj.p('t'), [NN 1], [1 1]);
            if t_range(1) > t_last || t_range(2) < t_first
                return;
            end
            has_tc = TEAStream.exists(obj.file_path, obj.p('t_coarse')) && ...
                TEAStream.exists(obj.file_path, obj.p('df_t_coarse'));
            if has_tc
                tc = double(h5read(obj.file_path, obj.p('t_coarse')));
                tc = tc(:);
                df = double(h5read(obj.file_path, obj.p('df_t_coarse')));
            else
                tc = [];
                df = 1;
            end
            s0 = obj.locate(t_range(1), tc, df, NN, 'start');
            s1 = obj.locate(t_range(2), tc, df, NN, 'end');
        end

        function idx = locate(obj, target, tc, df, NN, bound)
        % First sample with t >= target ('start') or last with t <= target ('end').
            if isempty(tc)
                lo = 1; hi = NN;
            else
                pos = @(k) (k - 1) * df + 1;   % sample index of t_coarse(k)
                if strcmp(bound, 'start')
                    k_lo = find(tc < target, 1, 'last');
                    k_hi = find(tc >= target, 1, 'first');
                else
                    k_lo = find(tc <= target, 1, 'last');
                    k_hi = find(tc > target, 1, 'first');
                end
                if isempty(k_lo), lo = 1; else, lo = pos(k_lo); end
                if isempty(k_hi), hi = NN; else, hi = min(pos(k_hi), NN); end
            end
            tw = h5read(obj.file_path, obj.p('t'), [lo 1], [hi - lo + 1, 1]);
            if strcmp(bound, 'start')
                r = find(tw >= target, 1, 'first');
                if isempty(r), idx = hi + 1; else, idx = lo + r - 1; end
            else
                r = find(tw <= target, 1, 'last');
                if isempty(r), idx = lo - 1; else, idx = lo + r - 1; end
            end
        end

        function di = disc_info_of(obj, t_out)
            NN = numel(t_out);
            di = struct('is_discontinuous', false, 'cont', zeros(0, 2), 'disc', zeros(0, 2));
            if NN > 0, di.cont = [1, NN]; end
            if NN < 2 || ~obj.isRegular || isempty(obj.rate) || obj.rate <= 0
                return;
            end
            gi = find(diff(t_out) > 1.1 / obj.rate);
            if ~isempty(gi)
                di.is_discontinuous = true;
                di.disc = [gi, gi + 1];
                di.cont = [[1; gi + 1], [gi; NN]];
            end
        end

    end % private methods

    methods (Static)

        function streams = open_all(file_path)
        % OPEN_ALL All streams of a TEA file, as a TEAStream array in file order.
            if TEAStream.schema_version_of(file_path) >= 2
                gi = h5info(file_path, '/rates');
                names = arrayfun(@(g) g.Name, gi.Groups, 'UniformOutput', false);
                k = cellfun(@(n) str2double(n(find(n == 'r', 1, 'last') + 1:end)), names);
                [~, order] = sort(k);
                streams = TEAStream.empty(1, 0);
                for j = 1:numel(order)
                    streams(j) = TEAStream(file_path, names{order(j)});
                end
            elseif TEAStream.exists(file_path, '/t') && TEAStream.exists(file_path, '/Samples')
                streams = TEAStream(file_path, '/');
            else
                streams = TEAStream.empty(1, 0);
            end
        end

        function v = schema_version_of(file_path)
        % SCHEMA_VERSION_OF Schema version stored in a file (1 when absent).
            v = 1;
            if TEAStream.exists(file_path, '/schema_version')
                v = round(double(h5read(file_path, '/schema_version')));
            end
        end

    end % static public

    methods (Static, Access = private)

        function tf = exists(file_path, path)
            fid = H5F.open(file_path, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
            c = onCleanup(@() H5F.close(fid));
            parts = strsplit(path(2:end), '/');
            tf = true;
            sofar = '';
            for k = 1:numel(parts)
                sofar = [sofar '/' parts{k}]; %#ok<AGROW>
                if ~H5L.exists(fid, sofar, 'H5P_DEFAULT')
                    tf = false;
                    return;
                end
            end
        end

        function sz = dims(file_path, path)
            info = h5info(file_path, path);
            sz = info.Dataspace.Size;   % MATLAB order, e.g. [N 1], [N C]
        end

        function v = read_scalar(file_path, path)
        % A MATLAB scalar, or [] when absent or stored empty.
            v = [];
            if ~TEAStream.exists(file_path, path), return; end
            try
                if h5readatt(file_path, path, 'MATLAB_empty'), return; end
            catch
            end
            v = h5read(file_path, path);
            if ~isa(v, 'int64'), v = double(v); end
        end

        function s = read_string(file_path, path)
            s = '';
            names = TEAStream.read_names(file_path, path);
            if ~isempty(names), s = [names{:}]; end
        end

        function names = read_names(file_path, path)
        % Rows of a char matrix dataset as a cell array, trailing blanks removed.
            names = {};
            if ~TEAStream.exists(file_path, path), return; end
            try
                x = h5read(file_path, path);
                if ~isnumeric(x) && ~ischar(x), return; end
                names = deblank(cellstr(char(x)))';
            catch
                names = {};
            end
        end

    end % static private

end
