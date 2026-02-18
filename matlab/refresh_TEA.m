function refresh_TEA(file_path)
% REFRESH_TEA Compute and add missing dependent variables to a TEA file.
%
%   refresh_TEA(file_path)
%
%   Opens an existing TEA file and recomputes all dependent variables:
%   t_coarse, df_t_coarse, isContinuous, cont, disc.
%
%   This is useful for:
%     - Files created without dependent variables
%     - Files that were modified outside of brew_TEA
%     - Upgrading legacy NRD/NSD .mat files (after adding isRegular)
%
%   Required variables must already exist: t, Samples, SR, isRegular
%   (SR can be empty if isRegular=false)
%
%   See also: brew_TEA, drink_TEA

    if ~exist(file_path, 'file')
        error('TEA:FileNotFound', 'File not found: %s', file_path);
    end

    mf = matfile(file_path, 'Writable', true);
    file_vars = whos(mf);
    var_names = {file_vars.name};

    % --- Check required variables ---
    required = {'t', 'Samples', 'isRegular'};
    for i = 1:length(required)
        if ~ismember(required{i}, var_names)
            error('TEA:MissingRequired', 'Required variable "%s" not found in %s.', required{i}, file_path);
        end
    end

    % SR is required for regular data
    isRegular = mf.isRegular;
    if ismember('SR', var_names)
        SR = mf.SR;
    else
        SR = [];
    end

    if isRegular && (isempty(SR) || SR <= 0)
        error('TEA:MissingSR', 'SR must be a positive scalar for regular data. File: %s', file_path);
    end

    % --- Load t ---
    t = mf.t(:, 1);
    N = length(t);

    % --- Compute t_coarse and df_t_coarse ---
    if isRegular && ~isempty(SR) && SR > 0
        df_t_coarse = round(SR);
    else
        df_t_coarse = max(1, round(N / max(1, (t(end) - t(1)))));
    end
    df_t_coarse = max(1, df_t_coarse);

    if N > 0
        t_coarse = t(1:df_t_coarse:end);
    else
        t_coarse = [];
    end

    mf.t_coarse = t_coarse;
    mf.df_t_coarse = df_t_coarse;
    fprintf('  Updated t_coarse (%d entries, df=%d)\n', length(t_coarse), df_t_coarse);

    % --- Compute isContinuous, cont, disc ---
    if isRegular && ~isempty(SR) && SR > 0 && N > 1
        dt = diff(t);
        gap_threshold = 1.1 / SR;
        gap_mask = dt > gap_threshold;

        if any(gap_mask)
            isContinuous = false;
            gap_indices = find(gap_mask);

            disc_arr = [gap_indices, gap_indices + 1];
            block_starts = [1; gap_indices + 1];
            block_stops = [gap_indices; N];
            cont_arr = [block_starts, block_stops];

            mf.disc = disc_arr;
            mf.cont = cont_arr;
            fprintf('  Discontinuous: %d gaps, %d continuous blocks\n', size(disc_arr, 1), size(cont_arr, 1));
        else
            isContinuous = true;
            fprintf('  Continuous: no gaps detected\n');
        end
    else
        isContinuous = true;
        fprintf('  Continuous (irregular or single sample)\n');
    end

    mf.isContinuous = isContinuous;

    % --- Add tea_version if missing ---
    if ~ismember('tea_version', var_names)
        mf.tea_version = '1.0';
        fprintf('  Added tea_version = 1.0\n');
    end

    fprintf('refresh_TEA complete: %s\n', file_path);

end
