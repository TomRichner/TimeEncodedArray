classdef test_tea < matlab.unittest.TestCase
% TEST_TEA Unit tests for the TEA class.
%
%   Run with: runtests('test_tea')

    properties
        temp_dir
    end

    methods (TestMethodSetup)
        function setup(tc)
            tc.temp_dir = fullfile(tempdir, 'tea_test');
            if ~exist(tc.temp_dir, 'dir')
                mkdir(tc.temp_dir);
            end
            addpath(fileparts(fileparts(mfilename('fullpath'))));
        end
    end

    methods (TestMethodTeardown)
        function teardown(tc)
            if exist(tc.temp_dir, 'dir')
                rmdir(tc.temp_dir, 's');
            end
        end
    end

    methods (Test)

        %% Test 1: Round-trip continuous
        function test_roundtrip_continuous(tc)
            f = fullfile(tc.temp_dir, 'cont.mat');
            SR = 1000; N = 5000;
            t = (0:N-1)' / SR;
            Samples = randn(N, 3);

            tea = TEA(f, SR, true, 't_units', 's');
            tea.write(t, Samples);

            [Data, t_out, disc_info] = tea.read([], [], []);
            tc.verifyEqual(t_out, t, 'AbsTol', 1e-12);
            tc.verifyEqual(Data, Samples, 'AbsTol', 1e-12);
            tc.verifyFalse(disc_info.is_discontinuous);
        end

        %% Test 2: Round-trip discontinuous
        function test_roundtrip_discontinuous(tc)
            f = fullfile(tc.temp_dir, 'disc.mat');
            SR = 1000;
            t1 = (0:999)' / SR;
            t2 = (3000:3999)' / SR;
            t = [t1; t2];
            Samples = randn(length(t), 2);

            tea = TEA(f, SR, true);
            tea.write(t, Samples);

            mf = matfile(f);
            tc.verifyFalse(mf.isContinuous);
            tc.verifyEqual(mf.disc, [1000, 1001]);
            tc.verifyEqual(mf.cont, [1, 1000; 1001, 2000]);

            [~, ~, di] = tea.read([], [], []);
            tc.verifyTrue(di.is_discontinuous);
        end

        %% Test 3: Sample range read
        function test_sample_range(tc)
            f = fullfile(tc.temp_dir, 'sr.mat');
            SR = 1000; N = 10000;
            t = (0:N-1)' / SR;
            Samples = (1:N)';

            tea = TEA(f, SR, true);
            tea.write(t, Samples);

            [Data, t_out] = tea.read([], [], [500, 1500]);
            tc.verifyEqual(length(t_out), 1001);
            tc.verifyEqual(Data(1), 500, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(end), 1500, 'AbsTol', 1e-12);
        end

        %% Test 4: Time range read
        function test_time_range(tc)
            f = fullfile(tc.temp_dir, 'tr.mat');
            SR = 1000; N = 10000;
            t = (0:N-1)' / SR;
            Samples = (1:N)';

            tea = TEA(f, SR, true);
            tea.write(t, Samples);

            [Data, t_out] = tea.read([], [1.0, 2.0], []);
            tc.verifyEqual(t_out(1), 1.0, 'AbsTol', 1e-6);
            tc.verifyEqual(t_out(end), 2.0, 'AbsTol', 1e-6);
            tc.verifyEqual(Data(1), 1001, 'AbsTol', 1e-12);
        end

        %% Test 5: Channel selection
        function test_channel_selection(tc)
            f = fullfile(tc.temp_dir, 'ch.mat');
            SR = 100; N = 500;
            t = (0:N-1)' / SR;
            Samples = [ones(N,1), 2*ones(N,1), 3*ones(N,1), 4*ones(N,1)];

            tea = TEA(f, SR, true);
            tea.write(t, Samples);
            % Write ch_map manually since write doesn't take it
            mf = matfile(f, 'Writable', true);
            mf.ch_map = [10 20 30 40];

            [Data, ~] = tea.read([20, 40], [], []);
            tc.verifyEqual(size(Data, 2), 2);
            tc.verifyEqual(Data(1, 1), 2, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(1, 2), 4, 'AbsTol', 1e-12);
        end

        %% Test 6: Channel default (no ch_map)
        function test_channel_default(tc)
            f = fullfile(tc.temp_dir, 'chd.mat');
            SR = 100; N = 500;
            t = (0:N-1)' / SR;
            Samples = [ones(N,1), 2*ones(N,1)];

            tea = TEA(f, SR, true);
            tea.write(t, Samples);

            [Data, ~] = tea.read([2], [], []);
            tc.verifyEqual(Data(1), 2, 'AbsTol', 1e-12);
        end

        %% Test 7: Append in time
        function test_append_time(tc)
            f = fullfile(tc.temp_dir, 'at.mat');
            SR = 1000; N = 5000;
            t1 = (0:N-1)' / SR;
            s1 = ones(N, 2);

            tea = TEA(f, SR, true);
            tea.write(t1, s1);

            t2 = t1(end) + (1:N)' / SR;
            s2 = 2 * ones(N, 2);
            tea.write(t2, s2);

            [Data, t_out] = tea.read([], [], []);
            tc.verifyEqual(length(t_out), 2*N);
            tc.verifyEqual(Data(1, 1), 1, 'AbsTol', 1e-12);
            tc.verifyEqual(Data(N+1, 1), 2, 'AbsTol', 1e-12);
            tc.verifyTrue(all(diff(t_out) > 0));
        end

        %% Test 8: Append channels
        function test_append_channels(tc)
            f = fullfile(tc.temp_dir, 'ac.mat');
            SR = 100; N = 500;
            t = (0:N-1)' / SR;
            s1 = ones(N, 2);

            tea = TEA(f, SR, true);
            tea.write(t, s1);
            tea.write_channels(3 * ones(N, 2), [3, 4]);

            [Data, ~] = tea.read([3], [], []);
            tc.verifyEqual(Data(1), 3, 'AbsTol', 1e-12);
            tc.verifyEqual(tea.C, 4);
        end

        %% Test 9: isRegular mismatch error
        function test_irregular_mismatch(tc)
            f = fullfile(tc.temp_dir, 'ie.mat');
            SR = 1000;
            t = cumsum([0; 0.001; 0.01; 0.1; 0.001; 0.5; 0.001; 0.3; 0.001; 0.2]);
            Samples = randn(length(t), 1);

            tea = TEA(f, SR, true);
            tc.verifyError(@() tea.write(t, Samples), 'TEA:IrregularData');
        end

        %% Test 10: refresh
        function test_refresh(tc)
            f = fullfile(tc.temp_dir, 'ref.mat');
            SR = 500; N = 5000;
            t = (0:N-1)' / SR;

            % Create minimal file
            dummy = 0; %#ok<NASGU>
            save(f, 'dummy', '-v7.3');
            mf = matfile(f, 'Writable', true);
            mf.t = t;
            mf.Samples = randn(N, 2);
            mf.SR = SR;
            mf.isRegular = true;

            tea = TEA(f, SR, true);
            tea.refresh();

            mf2 = matfile(f);
            vars = {whos(mf2).name};
            tc.verifyTrue(ismember('t_coarse', vars));
            tc.verifyTrue(mf2.isContinuous);
        end

        %% Test 11: Graceful degradation (no t_coarse)
        function test_no_tcoarse(tc)
            f = fullfile(tc.temp_dir, 'nc.mat');
            SR = 1000; N = 5000;
            t = (0:N-1)' / SR;

            dummy = 0; %#ok<NASGU>
            save(f, 'dummy', '-v7.3');
            mf = matfile(f, 'Writable', true);
            mf.t = t;
            mf.Samples = (1:N)';
            mf.SR = SR;
            mf.isRegular = true;
            mf.isContinuous = true;

            tea = TEA(f, SR, true);
            [~, t_out] = tea.read([], [1.0, 2.0], []);
            tc.verifyEqual(t_out(1), 1.0, 'AbsTol', 1e-6);
        end

        %% Test 12: Empty range
        function test_empty_range(tc)
            f = fullfile(tc.temp_dir, 'em.mat');
            SR = 1000; N = 1000;
            t = (0:N-1)' / SR;

            tea = TEA(f, SR, true);
            tea.write(t, randn(N, 1));

            [Data, t_out] = tea.read([], [100, 200], []);
            tc.verifyEmpty(Data);
            tc.verifyEmpty(t_out);
        end

        %% Test 13: Irregular data
        function test_irregular(tc)
            f = fullfile(tc.temp_dir, 'irr.mat');
            t = sort(rand(500, 1) * 100);
            Samples = randn(500, 2);

            tea = TEA(f, [], false);
            tea.write(t, Samples);

            [Data, t_out, di] = tea.read([], [], []);
            tc.verifyEqual(length(t_out), 500);
            tc.verifyFalse(di.is_discontinuous);
        end

        %% Test 14: info method
        function test_info(tc)
            f = fullfile(tc.temp_dir, 'inf.mat');
            SR = 1000;
            t = (0:999)' / SR;

            tea = TEA(f, SR, true, 't_units', 's');
            tea.write(t, randn(1000, 3));

            s = tea.info();
            tc.verifyEqual(s.N, 1000);
            tc.verifyEqual(s.C, 3);
            tc.verifyEqual(s.SR, 1000);
            tc.verifyTrue(s.isContinuous);
        end

        %% Test 15: Re-open existing file
        function test_reopen(tc)
            f = fullfile(tc.temp_dir, 'reopen.mat');
            SR = 1000;
            t = (0:999)' / SR;

            tea1 = TEA(f, SR, true);
            tea1.write(t, randn(1000, 2));

            % Re-open with matching params
            tea2 = TEA(f, SR, true);
            tc.verifyEqual(tea2.N, 1000);
            tc.verifyEqual(tea2.C, 2);

            % Append via new handle
            t2 = t(end) + (1:500)' / SR;
            tea2.write(t2, randn(500, 2));
            tc.verifyEqual(tea2.N, 1500);
        end

        %% Test 16: Append creates discontinuity at junction
        function test_append_creates_discontinuity(tc)
            f = fullfile(tc.temp_dir, 'jdisc.mat');
            SR = 1000;
            t1 = (0:999)' / SR;
            tea = TEA(f, SR, true);
            tea.write(t1, randn(1000, 2));

            % Append with a 2-second gap
            t2 = (3000:3999)' / SR;
            tea.write(t2, randn(1000, 2));

            mf = matfile(f);
            tc.verifyFalse(mf.isContinuous);
            tc.verifyEqual(mf.disc, [1000, 1001]);
            tc.verifyEqual(mf.cont, [1, 1000; 1001, 2000]);
        end

        %% Test 17: Append to already-discontinuous file
        function test_append_to_discontinuous_file(tc)
            f = fullfile(tc.temp_dir, 'adisc.mat');
            SR = 1000;

            % First write: continuous
            t1 = (0:999)' / SR;
            tea = TEA(f, SR, true);
            tea.write(t1, randn(1000, 1));

            % Second write: gap at junction
            t2 = (5000:5999)' / SR;
            tea.write(t2, randn(1000, 1));

            % Third write: another gap
            t3 = (9000:9999)' / SR;
            tea.write(t3, randn(1000, 1));

            mf = matfile(f);
            tc.verifyFalse(mf.isContinuous);
            tc.verifyEqual(size(mf.disc, 1), 2);
            tc.verifyEqual(mf.disc, [1000, 1001; 2000, 2001]);
            tc.verifyEqual(mf.cont, [1, 1000; 1001, 2000; 2001, 3000]);
        end

        %% Test 18: Append chunk with internal gaps
        function test_append_with_internal_gaps(tc)
            f = fullfile(tc.temp_dir, 'igap.mat');
            SR = 1000;
            t1 = (0:499)' / SR;
            tea = TEA(f, SR, true);
            tea.write(t1, randn(500, 1));

            % Append a chunk that itself has two internal gaps
            seg_a = (500:699)' / SR;     % continuous with t1
            seg_b = (3000:3199)' / SR;   % gap
            seg_c = (6000:6199)' / SR;   % gap
            t2 = [seg_a; seg_b; seg_c];
            tea.write(t2, randn(length(t2), 1));

            mf = matfile(f);
            tc.verifyFalse(mf.isContinuous);
            tc.verifyEqual(size(mf.disc, 1), 2);
            % Gaps at local indices 200 and 400 in t2 → global 700 and 900
            tc.verifyEqual(mf.disc, [700, 701; 900, 901]);
        end

        %% Test 19: Incremental matches full recompute
        function test_incremental_matches_full_recompute(tc)
            f = fullfile(tc.temp_dir, 'incr.mat');
            SR = 500;

            tea = TEA(f, SR, true);

            % Write, then append several chunks with mixed gaps
            t1 = (0:999)' / SR;
            tea.write(t1, randn(1000, 2));

            t2 = t1(end) + (1:500)' / SR;  % continuous
            tea.write(t2, randn(500, 2));

            t3 = (5000:5499)' / SR;  % gap
            tea.write(t3, randn(500, 2));

            % Snapshot incremental results
            mf = matfile(f);
            inc_cont = mf.cont;
            inc_disc = mf.disc;
            inc_isCont = mf.isContinuous;
            inc_tc = mf.t_coarse;

            % Full recompute
            tea.refresh();

            mf2 = matfile(f);
            tc.verifyEqual(inc_isCont, mf2.isContinuous);
            tc.verifyEqual(inc_disc, mf2.disc);
            tc.verifyEqual(inc_cont, mf2.cont);
            tc.verifyEqual(inc_tc, mf2.t_coarse, 'AbsTol', 1e-12);
        end

    end
end
