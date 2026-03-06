%% cross_validate_tea.m — Cross-validate Python and MATLAB TEA implementations
%
% Part 1: Read Python-written files and verify
% Part 2: Write MATLAB files for Python to read
%
% Run from TimeEncodedArray/matlab directory

function cross_validate_tea()

    % Add parent directory (matlab/) to path for TEA.m
    matlab_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    addpath(matlab_dir);
    
    cv_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'cross_validation_data');
    cv_dir = char(java.io.File(cv_dir).getCanonicalPath());
    
    fprintf('Cross-validation directory: %s\n', cv_dir);
    fprintf('============================================================\n');
    
    n_pass = 0;
    n_fail = 0;
    
    %% ============ PART 1: Read Python files ============
    fprintf('\nPART 1: Reading Python-written TEA files in MATLAB\n');
    fprintf('------------------------------------------------------------\n');
    
    % --- Case 1: Continuous ---
    fprintf('\nCase 1: py_continuous.mat\n');
    try
        f1 = fullfile(cv_dir, 'py_continuous.mat');
        mf = matfile(f1);
        
        % Check core variables exist
        vars = {whos(mf).name};
        assert(ismember('t', vars), 'Missing t');
        assert(ismember('Samples', vars), 'Missing Samples');
        assert(ismember('SR', vars), 'Missing SR');
        
        t = mf.t;
        S = mf.Samples;
        SR_val = mf.SR;
        
        % Verify dimensions
        assert(size(t, 1) == 5000, sprintf('t rows: expected 5000, got %d', size(t, 1)));
        assert(size(S, 1) == 5000, sprintf('Samples rows: expected 5000, got %d', size(S, 1)));
        assert(size(S, 2) == 3, sprintf('Samples cols: expected 3, got %d', size(S, 2)));
        assert(abs(SR_val - 1000) < 1e-10, sprintf('SR: expected 1000, got %g', SR_val));
        
        % Verify data values
        assert(abs(t(1) - 0) < 1e-10, 'First t should be 0');
        assert(abs(t(end) - 4.999) < 1e-6, sprintf('Last t: expected 4.999, got %g', t(end)));
        assert(all(S(:,1) == 1), 'Channel 1 should be all 1s');
        assert(all(S(:,2) == 2), 'Channel 2 should be all 2s');
        assert(all(S(:,3) == 3), 'Channel 3 should be all 3s');
        
        fprintf('  PASS\n');
        n_pass = n_pass + 1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
    
    % --- Case 2: Discontinuous ---
    fprintf('\nCase 2: py_discontinuous.mat\n');
    try
        f2 = fullfile(cv_dir, 'py_discontinuous.mat');
        mf = matfile(f2);
        
        t = mf.t;
        S = mf.Samples;
        
        assert(size(t, 1) == 2000, sprintf('N: expected 2000, got %d', size(t, 1)));
        assert(size(S, 2) == 1, sprintf('C: expected 1, got %d', size(S, 2)));
        
        % Verify data is sequential 1:2000
        expected = (1:2000)';
        assert(all(abs(S - expected) < 1e-10), 'Samples should be 1:2000');
        
        % Check gap exists: t(1000) should be ~1.998, t(1001) should be ~5.0
        assert(t(1001) - t(1000) > 2.0, 'Should have gap between samples 1000 and 1001');
        
        fprintf('  PASS\n');
        n_pass = n_pass + 1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
    
    % --- Case 3: Appended ---
    fprintf('\nCase 3: py_appended.mat\n');
    try
        f3 = fullfile(cv_dir, 'py_appended.mat');
        mf = matfile(f3);
        
        t = mf.t;
        S = mf.Samples;
        
        assert(size(t, 1) == 5000, sprintf('N: expected 5000, got %d', size(t, 1)));
        assert(size(S, 2) == 2, sprintf('C: expected 2, got %d', size(S, 2)));
        
        % First 3000 samples should be 10, last 2000 should be 20
        assert(all(S(1:3000, 1) == 10), 'First 3000 should be 10');
        assert(all(S(3001:5000, 1) == 20), 'Last 2000 should be 20');
        
        fprintf('  PASS\n');
        n_pass = n_pass + 1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
    
    % --- Case 4: Irregular ---
    fprintf('\nCase 4: py_irregular.mat\n');
    try
        f4 = fullfile(cv_dir, 'py_irregular.mat');
        mf = matfile(f4);
        
        t = mf.t;
        S = mf.Samples;
        
        assert(size(t, 1) == 100, sprintf('N: expected 100, got %d', size(t, 1)));
        assert(size(S, 2) == 1, sprintf('C: expected 1, got %d', size(S, 2)));
        
        % Data should be 1:100
        expected = (1:100)';
        assert(all(abs(S - expected) < 1e-10), 'Samples should be 1:100');
        
        % t should be monotonic
        assert(all(diff(t) > 0), 't should be monotonically increasing');
        
        fprintf('  PASS\n');
        n_pass = n_pass + 1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
    
    % --- Case 5: Open Python file with MATLAB TEA class ---
    fprintf('\nCase 5: Open py_continuous.mat via MATLAB TEA class\n');
    try
        f1 = fullfile(cv_dir, 'py_continuous.mat');
        tea = TEA(f1, 1000, true, 't_units', 's');
        
        assert(tea.N == 5000, sprintf('N: expected 5000, got %d', tea.N));
        assert(tea.C == 3, sprintf('C: expected 3, got %d', tea.C));
        
        [Data, t_out] = tea.read([1, 3], [], []);
        assert(size(Data, 2) == 2, 'Should read 2 channels');
        assert(all(Data(:,1) == 1), 'Channel 1 should be all 1s');
        assert(all(Data(:,2) == 3), 'Channel 3 should be all 3s');
        
        fprintf('  PASS\n');
        n_pass = n_pass + 1;
    catch ME
        fprintf('  FAIL: %s\n', ME.message);
        n_fail = n_fail + 1;
    end
    
    %% ============ PART 2: Write MATLAB files ============
    fprintf('\n============================================================\n');
    fprintf('PART 2: Writing MATLAB TEA files for Python to read\n');
    fprintf('------------------------------------------------------------\n');
    
    % --- MATLAB Case 1: Continuous ---
    fprintf('\nWriting ml_continuous.mat\n');
    f_ml1 = fullfile(cv_dir, 'ml_continuous.mat');
    if exist(f_ml1, 'file'), delete(f_ml1); end
    SR = 1000; N = 4000;
    t = (0:N-1)' / SR;
    S = [5*ones(N,1), 6*ones(N,1)];
    tea = TEA(f_ml1, SR, true, 't_units', 's');
    tea.write(t, S);
    fprintf('  Wrote: %s (N=%d, C=2)\n', f_ml1, N);
    
    % --- MATLAB Case 2: Discontinuous ---
    fprintf('Writing ml_discontinuous.mat\n');
    f_ml2 = fullfile(cv_dir, 'ml_discontinuous.mat');
    if exist(f_ml2, 'file'), delete(f_ml2); end
    SR = 500;
    t1 = (0:499)' / SR;
    t2 = 10 + (0:499)' / SR;
    t = [t1; t2];
    S = (1:1000)';
    tea = TEA(f_ml2, SR, true, 't_units', 's');
    tea.write(t, S);
    fprintf('  Wrote: %s (N=1000, C=1)\n', f_ml2);
    
    % --- MATLAB Case 3: With appended channels ---
    fprintf('Writing ml_channels.mat\n');
    f_ml3 = fullfile(cv_dir, 'ml_channels.mat');
    if exist(f_ml3, 'file'), delete(f_ml3); end
    SR = 1000; N = 2000;
    t = (0:N-1)' / SR;
    S = 7 * ones(N, 2);
    tea = TEA(f_ml3, SR, true);
    tea.write(t, S);
    tea.write_channels(8 * ones(N, 1), [3]);
    fprintf('  Wrote: %s (N=%d, C=3)\n', f_ml3, N);
    
    %% ============ SUMMARY ============
    fprintf('\n============================================================\n');
    fprintf('RESULTS: %d passed, %d failed\n', n_pass, n_fail);
    if n_fail == 0
        fprintf('ALL CROSS-VALIDATION TESTS PASSED\n');
    else
        fprintf('SOME TESTS FAILED\n');
    end
end
