%% write_and_compare_identical_tea.m
%
% 1. Write a MATLAB TEA file with the exact same data as write_identical_tea_py.py
% 2. Compare every variable in both files and report pass/fail
%
% Run from TimeEncodedArray/matlab/tests/

function write_and_compare_identical_tea()
    
    matlab_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    addpath(matlab_dir);
    
    cv_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'cross_validation_data');
    cv_dir = char(java.io.File(cv_dir).getCanonicalPath());
    
    %% ============ PART 1: Write MATLAB version ============
    fprintf('PART 1: Writing identical MATLAB TEA file\n');
    fprintf('============================================================\n');
    
    out = fullfile(cv_dir, 'identical_ml.mat');
    if exist(out, 'file'), delete(out); end
    
    SR = 500;
    tea = TEA(out, SR, true, 't_units', 's', 'tea_version', '1.0');
    
    % --- Write 1: initial chunk, 1000 samples, 3 channels ---
    t1 = (0:999)' / SR;
    s1 = [1*ones(1000,1), 2*ones(1000,1), 3*ones(1000,1)];
    tea.write(t1, s1);
    
    % --- Write 2: append with 2-second GAP, 500 samples ---
    t2 = t1(end) + 2.0 + (1:500)' / SR;
    s2 = [4*ones(500,1), 5*ones(500,1), 6*ones(500,1)];
    tea.write(t2, s2);
    
    % --- Write 3: append continuous, 500 samples ---
    t3 = t2(end) + (1:500)' / SR;
    s3 = [7*ones(500,1), 8*ones(500,1), 9*ones(500,1)];
    tea.write(t3, s3);
    
    fprintf('Wrote: %s\n\n', out);
    
    %% ============ PART 2: Compare ============
    fprintf('PART 2: Comparing Python vs MATLAB TEA files\n');
    fprintf('============================================================\n');
    
    py_file = fullfile(cv_dir, 'identical_py.mat');
    ml_file = fullfile(cv_dir, 'identical_ml.mat');
    
    if ~exist(py_file, 'file')
        error('Python file not found: %s\nRun write_identical_tea_py.py first.', py_file);
    end
    
    mf_py = matfile(py_file);
    mf_ml = matfile(ml_file);
    
    py_vars = sort({whos(mf_py).name});
    ml_vars = sort({whos(mf_ml).name});
    
    fprintf('\nPython vars: %s\n', strjoin(py_vars, ', '));
    fprintf('MATLAB vars: %s\n\n', strjoin(ml_vars, ', '));
    
    % Check that both have the same variables
    all_vars = union(py_vars, ml_vars);
    n_pass = 0;
    n_fail = 0;
    
    for i = 1:length(all_vars)
        var = all_vars{i};
        
        if ~ismember(var, py_vars)
            fprintf('  %-20s FAIL (missing in Python file)\n', var);
            n_fail = n_fail + 1;
            continue;
        end
        if ~ismember(var, ml_vars)
            fprintf('  %-20s FAIL (missing in MATLAB file)\n', var);
            n_fail = n_fail + 1;
            continue;
        end
        
        val_py = mf_py.(var);
        val_ml = mf_ml.(var);
        
        % Compare
        if ischar(val_py) && ischar(val_ml)
            if strcmp(val_py, val_ml)
                fprintf('  %-20s PASS (char: ''%s'')\n', var, val_py);
                n_pass = n_pass + 1;
            else
                fprintf('  %-20s FAIL (char: ''%s'' vs ''%s'')\n', var, val_py, val_ml);
                n_fail = n_fail + 1;
            end
        elseif islogical(val_py) && islogical(val_ml)
            if all(val_py == val_ml)
                fprintf('  %-20s PASS (logical: %d)\n', var, val_py);
                n_pass = n_pass + 1;
            else
                fprintf('  %-20s FAIL (logical: %d vs %d)\n', var, val_py, val_ml);
                n_fail = n_fail + 1;
            end
        elseif isnumeric(val_py) && isnumeric(val_ml)
            % Check sizes
            if ~isequal(size(val_py), size(val_ml))
                fprintf('  %-20s FAIL (size: %s vs %s)\n', var, mat2str(size(val_py)), mat2str(size(val_ml)));
                n_fail = n_fail + 1;
            else
                if isscalar(val_py)
                    if abs(val_py - val_ml) < 1e-12
                        fprintf('  %-20s PASS (scalar: %g)\n', var, val_py);
                        n_pass = n_pass + 1;
                    else
                        fprintf('  %-20s FAIL (scalar: %g vs %g)\n', var, val_py, val_ml);
                        n_fail = n_fail + 1;
                    end
                else
                    max_diff = max(abs(val_py(:) - val_ml(:)));
                    if max_diff < 1e-12
                        fprintf('  %-20s PASS (%s, max_diff=%.2e)\n', var, mat2str(size(val_py)), max_diff);
                        n_pass = n_pass + 1;
                    else
                        fprintf('  %-20s FAIL (%s, max_diff=%.2e)\n', var, mat2str(size(val_py)), max_diff);
                        n_fail = n_fail + 1;
                    end
                end
            end
        else
            fprintf('  %-20s SKIP (type mismatch: %s vs %s)\n', var, class(val_py), class(val_ml));
            n_fail = n_fail + 1;
        end
    end
    
    fprintf('\n============================================================\n');
    fprintf('RESULTS: %d passed, %d failed out of %d variables\n', n_pass, n_fail, length(all_vars));
    if n_fail == 0
        fprintf('ALL VARIABLES IDENTICAL\n');
    else
        fprintf('SOME VARIABLES DIFFER\n');
    end
end
