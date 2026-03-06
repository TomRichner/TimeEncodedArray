%% compare_test_cases.m — Compare Python vs MATLAB TEA files from MATLAB's perspective.
%
% For each case K=1..8, reads py_caseK.mat and ml_caseK.mat via matfile()
% and compares every variable for identical values.
%
% Tolerance: doubles < 1e-12, char/logical exact.

function compare_test_cases()

    matlab_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    addpath(matlab_dir);

    cv_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'cross_validation_data');
    cv_dir = char(java.io.File(cv_dir).getCanonicalPath());

    fprintf('MATLAB Comparator: Python vs MATLAB TEA files\n');
    fprintf('Directory: %s\n', cv_dir);
    fprintf('============================================================\n');

    total_pass = 0;
    total_fail = 0;

    for k = 1:8
        py_file = fullfile(cv_dir, sprintf('py_case%d.mat', k));
        ml_file = fullfile(cv_dir, sprintf('ml_case%d.mat', k));

        fprintf('\nCase %d:\n', k);

        if ~exist(py_file, 'file')
            fprintf('  SKIP — %s not found\n', sprintf('py_case%d.mat', k));
            continue;
        end
        if ~exist(ml_file, 'file')
            fprintf('  SKIP — %s not found\n', sprintf('ml_case%d.mat', k));
            continue;
        end

        [np, nf] = compare_files(py_file, ml_file);
        total_pass = total_pass + np;
        total_fail = total_fail + nf;
    end

    fprintf('\n============================================================\n');
    fprintf('TOTAL: %d passed, %d failed\n', total_pass, total_fail);
    if total_fail == 0
        fprintf('ALL IDENTICAL\n');
    else
        fprintf('SOME DIFFERENCES FOUND\n');
    end
end


function [n_pass, n_fail] = compare_files(py_file, ml_file)
    n_pass = 0;
    n_fail = 0;

    mf_py = matfile(py_file);
    mf_ml = matfile(ml_file);

    py_vars = sort({whos(mf_py).name});
    ml_vars = sort({whos(mf_ml).name});

    all_vars = union(py_vars, ml_vars);

    for i = 1:length(all_vars)
        var = all_vars{i};

        if ~ismember(var, py_vars)
            fprintf('  %-20s FAIL (missing in Python)\n', var);
            n_fail = n_fail + 1;
            continue;
        end
        if ~ismember(var, ml_vars)
            fprintf('  %-20s FAIL (missing in MATLAB)\n', var);
            n_fail = n_fail + 1;
            continue;
        end

        val_py = mf_py.(var);
        val_ml = mf_ml.(var);

        % Char comparison
        if ischar(val_py) && ischar(val_ml)
            if strcmp(val_py, val_ml)
                fprintf('  %-20s PASS (char)\n', var);
                n_pass = n_pass + 1;
            else
                fprintf('  %-20s FAIL (char: ''%s'' vs ''%s'')\n', var, val_py, val_ml);
                n_fail = n_fail + 1;
            end
            continue;
        end

        % Logical comparison (exact)
        if islogical(val_py) && islogical(val_ml)
            if isequal(val_py, val_ml)
                fprintf('  %-20s PASS (logical)\n', var);
                n_pass = n_pass + 1;
            else
                fprintf('  %-20s FAIL (logical: %d vs %d)\n', var, val_py, val_ml);
                n_fail = n_fail + 1;
            end
            continue;
        end

        % Numeric comparison
        if isnumeric(val_py) && isnumeric(val_ml)
            if ~isequal(size(val_py), size(val_ml))
                fprintf('  %-20s FAIL (size: %s vs %s)\n', var, mat2str(size(val_py)), mat2str(size(val_ml)));
                n_fail = n_fail + 1;
                continue;
            end

            if isempty(val_py) && isempty(val_ml)
                fprintf('  %-20s PASS (empty %s)\n', var, mat2str(size(val_py)));
                n_pass = n_pass + 1;
                continue;
            end

            max_diff = max(abs(double(val_py(:)) - double(val_ml(:))));
            if max_diff < 1e-12
                fprintf('  %-20s PASS (%s, max_diff=%.2e)\n', var, mat2str(size(val_py)), max_diff);
                n_pass = n_pass + 1;
            else
                fprintf('  %-20s FAIL (%s, max_diff=%.2e)\n', var, mat2str(size(val_py)), max_diff);
                n_fail = n_fail + 1;
            end
            continue;
        end

        % Type mismatch
        fprintf('  %-20s FAIL (type: %s vs %s)\n', var, class(val_py), class(val_ml));
        n_fail = n_fail + 1;
    end
end
