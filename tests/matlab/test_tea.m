classdef test_tea < matlab.unittest.TestCase
% TEST_TEA Unit tests for the TEA (Time-Encoded Array) MATLAB library.
%
%   Run with: runtests('test_tea')

    properties
        temp_dir
        tea_matlab_dir
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.temp_dir = fullfile(tempdir, 'tea_test');
            if ~exist(tc.temp_dir, 'dir')
                mkdir(tc.temp_dir);
            end
            % Add TEA matlab dir to path
            tc.tea_matlab_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'matlab');
            addpath(tc.tea_matlab_dir);
        end
    end

    methods (TestMethodTeardown)
        function teardown(tc)
            % Clean up temp files
            if exist(tc.temp_dir, 'dir')
                rmdir(tc.temp_dir, 's');
            end
        end
    end

    methods (Test)

        %% Test 1: Round-trip continuous
        function test_roundtrip_continuous(tc)
            f = fullfile(tc.temp_dir, 'cont.mat');
            SR = 1000;
            N = 5000;
            t = (0:N-1)' / SR;
            Samples = randn(N, 3);

            brew_TEA(f, t, Samples, SR, true, 't_units', 's', 'ch_map', [1 2 3]);

            [Data, t_out, disc_info] = drink_TEA(f, [], [], []);

            tc.verifyEqual(t_out, t, 'AbsTol', 1e-12);
            tc.verifyEqual(Data, Samples, 'AbsTol', 1e-12);
            tc.verifyFalse(disc_info.is_discontinuous);
        end

        %% Test 2: Round-trip discontinuous
        function test_roundtrip_discontinuous(tc)
            f = fullfile(tc.temp_dir, 'disc.mat');
            SR = 1000;
            t1 = (0:999)' / SR;
            t2 = (3000:3999)' / SR; % 2 second gap
            t = [t1; t2];
            Samples = randn(length(t), 2);

            brew_TEA(f, t, Samples, SR, true, 't_units', 's');

            % Check file-level variables
            mf = matfile(f);
            tc.verifyFalse(mf.isContinuous);
            tc.verifyEqual(size(mf.disc), [1, 2]);
            tc.verifyEqual(mf.disc, [1000, 1001]);
            tc.verifyEqual(size(mf.cont), [2, 2]);
            tc.verifyEqual(mf.cont, [1, 1000; 1001, 2000]);

            % Check disc_info from reader
            [~, ~, disc_info] = drink_TEA(f, [], [], []);
            tc.verifyTrue(disc_info.is_discontinuous);
            tc.verifyEqual(disc_info.disc, [1000, 1001]);
            tc.verifyEqual(disc_info.cont, [1, 1000; 1001, 2000]);
        end

        %% Test 3: Sample range read
        function test_sample_range(tc)
            f = fullfile(tc.temp_dir, 'srange.mat');
            SR = 1000;
            N = 10000;
            t = (0:N-1)' / SR;
            Samples = (1:N)';

            brew_TEA(f, t, Samples, SR, true);

            [Data, t_out] = drink_TEA(f, [], [], [500, 1500]);
            tc.verifyEqual(length(t_out), 1001);
            tc.verifyEqual(Data(1), 500, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(end), 1500, 'AbsTol', 1e-12);
            tc.verifyEqual(t_out(1), t(500), 'AbsTol', 1e-12);
        end

        %% Test 4: Time range read
        function test_time_range(tc)
            f = fullfile(tc.temp_dir, 'trange.mat');
            SR = 1000;
            N = 10000;
            t = (0:N-1)' / SR;
            Samples = (1:N)';

            brew_TEA(f, t, Samples, SR, true);

            [Data, t_out] = drink_TEA(f, [], [1.0, 2.0], []);
            % t=1.0 is sample 1001, t=2.0 is sample 2001
            tc.verifyEqual(t_out(1), 1.0, 'AbsTol', 1e-6);
            tc.verifyEqual(t_out(end), 2.0, 'AbsTol', 1e-6);
            tc.verifyEqual(Data(1), 1001, 'AbsTol', 1e-12);
        end

        %% Test 5: Channel selection
        function test_channel_selection(tc)
            f = fullfile(tc.temp_dir, 'chsel.mat');
            SR = 100;
            N = 500;
            t = (0:N-1)' / SR;
            Samples = [ones(N,1), 2*ones(N,1), 3*ones(N,1), 4*ones(N,1)];

            brew_TEA(f, t, Samples, SR, true, 'ch_map', [10 20 30 40]);

            [Data, ~] = drink_TEA(f, [20, 40], [], []);
            tc.verifyEqual(size(Data, 2), 2);
            tc.verifyEqual(Data(1, 1), 2, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(1, 2), 4, 'AbsTol', 1e-12);
        end

        %% Test 6: Channel default (no ch_map)
        function test_channel_default(tc)
            f = fullfile(tc.temp_dir, 'chdef.mat');
            SR = 100;
            N = 500;
            t = (0:N-1)' / SR;
            Samples = [ones(N,1), 2*ones(N,1)];

            brew_TEA(f, t, Samples, SR, true); % no ch_map

            [Data, ~] = drink_TEA(f, [2], [], []);
            tc.verifyEqual(size(Data, 2), 1);
            tc.verifyEqual(Data(1), 2, 'AbsTol', 1e-12);
        end

        %% Test 7: Append in time
        function test_append_time(tc)
            f = fullfile(tc.temp_dir, 'append_t.mat');
            SR = 1000;
            N = 5000;
            t1 = (0:N-1)' / SR;
            s1 = ones(N, 2);

            brew_TEA(f, t1, s1, SR, true);

            t2 = t1(end) + (1:N)' / SR;
            s2 = 2 * ones(N, 2);
            brew_TEA(f, t2, s2, SR, true, 'mode', 'append_time');

            [Data, t_out] = drink_TEA(f, [], [], []);
            tc.verifyEqual(length(t_out), 2*N);
            tc.verifyEqual(Data(1, 1), 1, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(N+1, 1), 2, 'AbsTol', 1e-12);
            % Verify monotonicity
            tc.verifyTrue(all(diff(t_out) > 0));
        end

        %% Test 8: Append channels
        function test_append_channels(tc)
            f = fullfile(tc.temp_dir, 'append_ch.mat');
            SR = 100;
            N = 500;
            t = (0:N-1)' / SR;
            s1 = ones(N, 2);

            brew_TEA(f, t, s1, SR, true, 'ch_map', [1, 2]);

            s2 = 3 * ones(N, 2);
            brew_TEA(f, [], s2, SR, true, 'mode', 'append_channels', 'ch_map', [3, 4]);

            [Data, ~] = drink_TEA(f, [3], [], []);
            tc.verifyEqual(Data(1), 3, 'AbsTol', 1e-12);

            [Data_all, ~] = drink_TEA(f, [], [], []);
            tc.verifyEqual(size(Data_all, 2), 4);
        end

        %% Test 9: isRegular mismatch error
        function test_irregular_mismatch(tc)
            f = fullfile(tc.temp_dir, 'irreg_err.mat');
            SR = 1000;
            % Create timestamps with wildly varying intervals
            % e.g., [0, 0.001, 0.01, 0.1, 0.2, 0.25, ...]
            t = cumsum([0; 0.001; 0.01; 0.1; 0.001; 0.5; 0.001; 0.3; 0.001; 0.2]);
            Samples = randn(length(t), 1);

            % Claim regular but data is clearly irregular -> should error
            tc.verifyError(@() brew_TEA(f, t, Samples, SR, true), 'TEA:IrregularData');
        end

        %% Test 10: refresh_TEA
        function test_refresh(tc)
            f = fullfile(tc.temp_dir, 'refresh.mat');
            SR = 500;
            N = 5000;
            t = (0:N-1)' / SR;
            Samples = randn(N, 2);

            % Manually create a minimal file without dependents
            dummy = 0; %#ok<NASGU>
            save(f, 'dummy', '-v7.3');
            mf = matfile(f, 'Writable', true);
            mf.t = t;
            mf.Samples = Samples;
            mf.SR = SR;
            mf.isRegular = true;

            % Verify dependents are missing
            vars_before = whos(mf);
            tc.verifyFalse(ismember('t_coarse', {vars_before.name}));

            refresh_TEA(f);

            mf2 = matfile(f);
            vars_after = whos(mf2);
            tc.verifyTrue(ismember('t_coarse', {vars_after.name}));
            tc.verifyTrue(ismember('isContinuous', {vars_after.name}));
            tc.verifyTrue(mf2.isContinuous);
        end

        %% Test 11: Graceful degradation (no t_coarse)
        function test_no_tcoarse(tc)
            f = fullfile(tc.temp_dir, 'no_coarse.mat');
            SR = 1000;
            N = 5000;
            t = (0:N-1)' / SR;
            Samples = (1:N)';

            brew_TEA(f, t, Samples, SR, true);

            % Delete t_coarse and df_t_coarse
            % We'll create a new file without them
            f2 = fullfile(tc.temp_dir, 'no_coarse2.mat');
            dummy = 0; %#ok<NASGU>
            save(f2, 'dummy', '-v7.3');
            mf = matfile(f2, 'Writable', true);
            mf.t = t;
            mf.Samples = Samples;
            mf.SR = SR;
            mf.isRegular = true;
            mf.isContinuous = true;

            % Read by time range should still work (with warning)
            [Data, t_out] = drink_TEA(f2, [], [1.0, 2.0], []);
            tc.verifyEqual(t_out(1), 1.0, 'AbsTol', 1e-6);
        end

        %% Test 12: Empty range
        function test_empty_range(tc)
            f = fullfile(tc.temp_dir, 'empty.mat');
            SR = 1000;
            N = 1000;
            t = (0:N-1)' / SR;
            Samples = randn(N, 1);

            brew_TEA(f, t, Samples, SR, true);

            % Request time range completely outside data
            [Data, t_out] = drink_TEA(f, [], [100, 200], []);
            tc.verifyEmpty(Data);
            tc.verifyEmpty(t_out);
        end

        %% Test 13: Irregular data
        function test_irregular(tc)
            f = fullfile(tc.temp_dir, 'irreg.mat');
            t = sort(rand(500, 1) * 100); % random timestamps
            Samples = randn(500, 2);

            brew_TEA(f, t, Samples, [], false);

            [Data, t_out, disc_info] = drink_TEA(f, [], [], []);
            tc.verifyEqual(length(t_out), 500);
            tc.verifyFalse(disc_info.is_discontinuous);

            mf = matfile(f);
            tc.verifyTrue(mf.isContinuous);
        end

    end
end
