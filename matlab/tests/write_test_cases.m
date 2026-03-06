%% write_test_cases.m — Write 9 deterministic TEA test files.
%
% Each case uses hardcoded data identical to write_test_cases.py.
% Output: cross_validation_data/ml_case{1..9}.mat
%
% Run from TimeEncodedArray/matlab/tests/

function write_test_cases()

    matlab_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    addpath(matlab_dir);

    cv_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'cross_validation_data');
    cv_dir = char(java.io.File(cv_dir).getCanonicalPath());
    if ~exist(cv_dir, 'dir'), mkdir(cv_dir); end

    case1(cv_dir);
    case2(cv_dir);
    case3(cv_dir);
    case4(cv_dir);
    case5(cv_dir);
    case6(cv_dir);
    case7(cv_dir);
    case8(cv_dir);
    case9(cv_dir);

    fprintf('\nAll 9 MATLAB test cases written to: %s\n', cv_dir);
end


function case1(cv_dir)
    %% Case 1: Write 5000x3 continuous, SR=1000
    f = fullfile(cv_dir, 'ml_case1.mat');
    if exist(f,'file'), delete(f); end
    SR = 1000; N = 5000;
    t = (0:N-1)' / SR;
    s = [1*ones(N,1), 2*ones(N,1), 3*ones(N,1)];
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t, s);
    fprintf('  Case 1: ml_case1.mat\n');
end


function case2(cv_dir)
    %% Case 2: Write 2000x1 with 1 gap, SR=500
    f = fullfile(cv_dir, 'ml_case2.mat');
    if exist(f,'file'), delete(f); end
    SR = 500;
    t1 = (0:999)' / SR;
    t2 = 5.0 + (0:999)' / SR;
    t = [t1; t2];
    s = (1:2000)';
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t, s);
    fprintf('  Case 2: ml_case2.mat\n');
end


function case3(cv_dir)
    %% Case 3: Write 1000x2, append 500x2 continuous, SR=1000
    f = fullfile(cv_dir, 'ml_case3.mat');
    if exist(f,'file'), delete(f); end
    SR = 1000;
    t1 = (0:999)' / SR;
    s1 = [10*ones(1000,1), 20*ones(1000,1)];
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t1, s1);

    t2 = t1(end) + (1:500)' / SR;
    s2 = [30*ones(500,1), 40*ones(500,1)];
    tea.write(t2, s2);
    fprintf('  Case 3: ml_case3.mat\n');
end


function case4(cv_dir)
    %% Case 4: Write 1000x2, append 500x2 with 3-sec gap, SR=1000
    f = fullfile(cv_dir, 'ml_case4.mat');
    if exist(f,'file'), delete(f); end
    SR = 1000;
    t1 = (0:999)' / SR;
    s1 = [10*ones(1000,1), 20*ones(1000,1)];
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t1, s1);

    t2 = t1(end) + 3.0 + (1:500)' / SR;
    s2 = [30*ones(500,1), 40*ones(500,1)];
    tea.write(t2, s2);
    fprintf('  Case 4: ml_case4.mat\n');
end


function case5(cv_dir)
    %% Case 5: Write 2000x2, write_channels 2000x1, SR=1000
    f = fullfile(cv_dir, 'ml_case5.mat');
    if exist(f,'file'), delete(f); end
    SR = 1000; N = 2000;
    t = (0:N-1)' / SR;
    s = [5*ones(N,1), 6*ones(N,1)];
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t, s);
    tea.write_channels(7*ones(N,1), [3]);
    fprintf('  Case 5: ml_case5.mat\n');
end


function case6(cv_dir)
    %% Case 6: Write 3000x1 with 3 gaps (4 segments of 750), SR=500
    f = fullfile(cv_dir, 'ml_case6.mat');
    if exist(f,'file'), delete(f); end
    SR = 500;
    segs = [];
    for i = 0:3
        offset = i * (750 / SR + 2.0);
        seg = offset + (0:749)' / SR;
        segs = [segs; seg]; %#ok<AGROW>
    end
    s = (1:3000)';
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(segs, s);
    fprintf('  Case 6: ml_case6.mat\n');
end


function case7(cv_dir)
    %% Case 7: Write 100x1 irregular (no SR)
    f = fullfile(cv_dir, 'ml_case7.mat');
    if exist(f,'file'), delete(f); end
    % t(k) = ((k-1)/99)^1.5 * 10, for k=1:100 (matches Python k=0:99)
    t = ((0:99)' / 99) .^ 1.5 * 10;
    s = (1:100)';
    tea = TEA(f, [], false);
    tea.write(t, s);
    fprintf('  Case 7: ml_case7.mat\n');
end


function case8(cv_dir)
    %% Case 8: Write 1000x2 + append 500x2 with gap + append 500x2 continuous, SR=500
    f = fullfile(cv_dir, 'ml_case8.mat');
    if exist(f,'file'), delete(f); end
    SR = 500;
    t1 = (0:999)' / SR;
    s1 = [1*ones(1000,1), 2*ones(1000,1)];
    tea = TEA(f, SR, true, 't_units', 's');
    tea.write(t1, s1);

    % Append with 2-sec gap
    t2 = t1(end) + 2.0 + (1:500)' / SR;
    s2 = [3*ones(500,1), 4*ones(500,1)];
    tea.write(t2, s2);

    % Append continuous
    t3 = t2(end) + (1:500)' / SR;
    s3 = [5*ones(500,1), 6*ones(500,1)];
    tea.write(t3, s3);
    fprintf('  Case 8: ml_case8.mat\n');
end


function case9(cv_dir)
    %% Case 9: Write 1000x1 with t_offset, SR=1000
    f = fullfile(cv_dir, 'ml_case9.mat');
    if exist(f,'file'), delete(f); end
    SR = 1000; N = 1000;
    t = (0:N-1)' / SR;
    s = (1:N)';
    tea = TEA(f, SR, true, 't_units', 's', ...
        't_offset', int64(1770000000), ...
        't_offset_units', 'posix_s', ...
        't_offset_scale', 1.0);
    tea.write(t, s);
    fprintf('  Case 9: ml_case9.mat\n');
end
