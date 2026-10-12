classdef test_tea_v2 < matlab.unittest.TestCase
% TEST_TEA_V2 Tests for TEA schema version 2 (several sample rates per file)
% and for reading version-1 files through TEA.rates / TEA.stream.
%
%   Run with: runtests('test_tea_v2')
%
%   Version-2 files are written by Python: python/tests/write_v2_test_cases.py
%   writes them to cross_validation_data/v2. When they are missing, the class
%   setup runs that script through `uv run`; if that fails the tests that need
%   them are skipped (assumption failure), not failed.

    properties
        data_dir
        temp_dir
        have_v2 = false
    end

    properties (Constant)
        T_OFFSET = 1770000000
    end

    methods (TestClassSetup)
        function find_v2_files(tc)
            here = fileparts(mfilename('fullpath'));
            matlab_dir = fileparts(here);
            repo = fileparts(matlab_dir);
            addpath(matlab_dir);
            tc.data_dir = fullfile(repo, 'cross_validation_data', 'v2');
            needed = {'v2_two_rates.mat', 'v2_two_rates_compressed.mat', 'v2_one_rate.mat', 'v1_python.mat'};
            have = @() all(cellfun(@(f) exist(fullfile(tc.data_dir, f), 'file') == 2, needed));
            if ~have()
                cmd = sprintf('uv run --directory "%s" python python/tests/write_v2_test_cases.py', repo);
                [status, out] = system(cmd);
                if status ~= 0
                    fprintf('Could not write the v2 test files:\n%s\n', out);
                end
            end
            tc.have_v2 = have();
        end
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.temp_dir = tempname;
            mkdir(tc.temp_dir);
        end
    end

    methods (TestMethodTeardown)
        function teardown(tc)
            if exist(tc.temp_dir, 'dir'), rmdir(tc.temp_dir, 's'); end
        end
    end

    methods (Access = private)
        function f = v2file(tc, name)
            tc.assumeTrue(tc.have_v2, 'Version-2 test files are missing (run write_v2_test_cases.py).');
            f = fullfile(tc.data_dir, name);
        end
    end

    methods (Static, Access = private)
        function x = ramp(n, c, scale)
            if nargin < 3, scale = 1; end
            x = reshape(0:n*c-1, c, n)' * scale;
        end
        function [t256, x256, t1024, x1024] = two_rate_data()
            t256 = (0:1023)' / 256;
            x256 = test_tea_v2.ramp(1024, 3);
            t1024 = [(0:1535)' / 1024; 3.5 + (0:2559)' / 1024];
            x1024 = test_tea_v2.ramp(4096, 2, -0.5);
        end
    end

    methods (Test)

        %% Version-1 files still read (MATLAB-written)
        function test_v1_matlab_file_as_stream(tc)
            f = fullfile(tc.temp_dir, 'v1.mat');
            t = (0:2999)' / 500;
            x = randn(3000, 2);
            tea = TEA(f, 500, true, 't_units', 's');
            tea.write(t, x);
            tc.verifyEqual(tea.schema_version, 1);
            tc.verifyEqual(tea.rates, 500);
            st = tea.stream(500);
            tc.verifyEqual(st.group, '/');
            tc.verifyEqual(st.channel_names, {'Ch1', 'Ch2'});
            [D, t_out] = st.read([], [], []);
            tc.verifyEqual(D, x);
            tc.verifyEqual(t_out, t);
            [D, t_out] = st.read([2], [1.0, 1.5], []);
            sel = t >= 1.0 & t <= 1.5;
            tc.verifyEqual(t_out, t(sel));
            tc.verifyEqual(D, x(sel, 2));
            % Version-1 read unchanged
            [D1, t1] = tea.read([], [], []);
            tc.verifyEqual(D1, x);
            tc.verifyEqual(t1, t);
        end

        %% Version-1 file written by Python, channel names from hdr
        function test_v1_python_file(tc)
            f = tc.v2file('v1_python.mat');
            tea = TEA(f, 500, true);
            tc.verifyEqual(tea.schema_version, 1);
            st = tea.stream();
            tc.verifyEqual(st.channel_names, {'A1', 'Bx2'});
            [D, t] = st.read({'Bx2'}, [], [101, 110]);
            x = test_tea_v2.ramp(3000, 2);
            tc.verifyEqual(D, x(101:110, 2));
            tc.verifyEqual(t, (100:109)' / 500, 'AbsTol', 1e-12);
        end

        %% Version-2 round trip with two rates
        function test_two_rates(tc)
            f = tc.v2file('v2_two_rates.mat');
            [t256, x256, t1024, x1024] = test_tea_v2.two_rate_data();
            tea = TEA(f, 256, true);
            tc.verifyEqual(tea.schema_version, 2);
            tc.verifyEqual(tea.rates, [256 1024]);

            s1 = tea.stream(256);
            tc.verifyEqual(s1.channel_names, {'Fp1', 'Fp2', 'Cz'});
            tc.verifyEqual(s1.units, {'uV', 'uV', 'uV'});
            tc.verifyEqual(s1.N, 1024);
            tc.verifyEqual(s1.C, 3);
            tc.verifyEqual(s1.t_offset, tc.T_OFFSET);
            tc.verifyEqual(s1.t_offset_units, 'posix_s');
            tc.verifyEqual(s1.t_units, 's');
            [D, t] = s1.read();
            tc.verifyEqual(D, x256);
            tc.verifyEqual(t, t256, 'AbsTol', 1e-12);

            s2 = tea.stream(1024);
            tc.verifyEqual(s2.channel_names, {'DC1', 'DC2'});
            tc.verifyEqual(s2.units, {'mV', 'uV'});
            [D, t] = s2.read([], [], []);
            tc.verifyEqual(D, x1024);
            tc.verifyEqual(t, t1024, 'AbsTol', 1e-12);

            % Whole-file load gives the struct layout
            S = load(f);
            tc.verifyEqual(S.schema_version, 2);
            tc.verifyEqual(cellstr(S.channel_names)', {'Fp1', 'Fp2', 'Cz', 'DC1', 'DC2'});
            tc.verifyEqual(S.channel_rates, [256; 256; 256; 1024; 1024]);
            tc.verifyEqual(S.rates.r2.Samples, x1024);

            % Version-1 read on a multi-rate file explains itself
            tc.verifyError(@() tea.read(), 'TEA:MultiRate');
            tc.verifyError(@() tea.stream(500), 'TEA:RateNotFound');
            tc.verifyError(@() tea.stream(), 'TEA:RateRequired');
        end

        %% A gap inside one rate is preserved
        function test_gap_inside_rate(tc)
            f = tc.v2file('v2_two_rates.mat');
            [~, ~, t1024, x1024] = test_tea_v2.two_rate_data();
            s2 = TEA(f, 1024, true).stream(1024);
            [~, t, di] = s2.read([], [], []);
            tc.verifyEqual(t, t1024, 'AbsTol', 1e-12);
            tc.verifyTrue(di.is_discontinuous);
            tc.verifyEqual(di.disc, [1536, 1537]);
            tc.verifyEqual(di.cont, [1, 1536; 1537, 4096]);
            S = load(f);
            tc.verifyFalse(S.rates.r2.isContinuous);
            tc.verifyEqual(S.rates.r2.disc, [1536, 1537]);
            tc.verifyTrue(S.rates.r1.isContinuous);
            % Window inside the gap is empty; one spanning it has both sides
            [D, t] = s2.read([], [2.0, 3.0], []);
            tc.verifyEmpty(D);
            tc.verifyEmpty(t);
            [D, t, di] = s2.read([], [1.4, 3.6], []);
            sel = t1024 >= 1.4 & t1024 <= 3.6;
            tc.verifyEqual(t, t1024(sel), 'AbsTol', 1e-12);
            tc.verifyEqual(D, x1024(sel, :));
            tc.verifyTrue(di.is_discontinuous);
        end

        %% Chunked partial reads, compressed file
        function test_partial_reads_compressed(tc)
            f = tc.v2file('v2_two_rates_compressed.mat');
            [t256, x256, t1024, x1024] = test_tea_v2.two_rate_data();
            info = h5info(f, '/rates/r2/Samples');
            tc.verifyEqual(info.ChunkSize, [TEA.chunk_samples(2, 'double') 2]);
            tc.verifyEqual(TEA.chunk_samples(2, 'double'), 8192);
            tc.verifyEqual(TEA.chunk_samples(276, 'double'), 237);
            tc.verifyEqual(TEA.chunk_samples(276, 'int32'), 474);
            names = {info.Filters.Name};
            tc.verifyTrue(any(strcmpi(names, 'deflate')));
            tc.verifyFalse(any(strcmpi(names, 'shuffle')));

            tea = TEA(f, 256, true);
            % Sample range across chunk boundaries
            [D, t] = tea.stream(256).read([], [], [480, 1020]);
            tc.verifyEqual(D, x256(480:1020, :));
            tc.verifyEqual(t, t256(480:1020), 'AbsTol', 1e-12);
            % Time range with channels by name and number, out of order
            [D, t] = tea.stream(1024).read({'DC2', 1}, [0.4, 0.9], []);
            sel = t1024 >= 0.4 & t1024 <= 0.9;
            tc.verifyEqual(t, t1024(sel), 'AbsTol', 1e-12);
            tc.verifyEqual(D, x1024(sel, [2 1]));
            [D, t] = tea.stream(1024).read(["DC1"], [3.9, 4.2], []);
            sel = t1024 >= 3.9 & t1024 <= 4.2;
            tc.verifyEqual(t, t1024(sel), 'AbsTol', 1e-12);
            tc.verifyEqual(D, x1024(sel, 1));
        end

        %% A one-rate version-2 file behaves exactly like version 1
        function test_one_rate_like_v1(tc)
            f = tc.v2file('v2_one_rate.mat');
            t = [(0:899)' / 200; 10 + (0:599)' / 200];
            x = test_tea_v2.ramp(1500, 4);
            tea = TEA(f, 200, true);
            tc.verifyEqual(tea.N, 1500);
            tc.verifyEqual(tea.C, 4);
            [D, t_out, di] = tea.read([], [], []);
            tc.verifyEqual(D, x);
            tc.verifyEqual(t_out, t, 'AbsTol', 1e-12);
            tc.verifyTrue(di.is_discontinuous);
            [D, t_out] = tea.read([3], [10.0, 10.5], []);
            sel = t >= 10.0 & t <= 10.5;
            tc.verifyEqual(t_out, t(sel), 'AbsTol', 1e-12);
            tc.verifyEqual(D, x(sel, 3));
            s = tea.info();
            tc.verifyFalse(s.isContinuous);
            mf = matfile(f);
            tc.verifyEqual(mf.t_offset, tc.T_OFFSET);
            % Same data through the version-2 API
            tc.verifyEqual(tea.rates, 200);
            st = tea.stream(200);
            tc.verifyEqual(st.channel_names, {'a', 'b', 'c', 'd'});
            [D2, t2] = st.read();
            tc.verifyEqual(D2, x);
            tc.verifyEqual(t2, t, 'AbsTol', 1e-12);
            % Version-1 writes are refused on a version-2 file
            g = fullfile(tc.temp_dir, 'copy.mat');
            copyfile(f, g);
            tea2 = TEA(g, 200, true);
            tc.verifyError(@() tea2.write(20 + (0:9)' / 200, zeros(10, 4)), 'TEA:Version2File');
        end

    end
end
