function initialize_savefast_chunk_4_column(filename, var, sz, var_class)
% INITIALIZE_SAVEFAST_CHUNK_4_COLUMN Initialize an HDF5 .mat variable with chunked storage.
%
%   initialize_savefast_chunk_4_column(filename, var, sz, var_class)
%
%   Creates or appends a chunked HDF5 dataset to a v7.3 .mat file.
%   Datasets are created with unlimited max dimensions (Inf) so they
%   can be resized later for append operations.
%   Can be called multiple times to initialize multiple variables.
%
%   Inputs:
%       filename  - Path to the .mat file (created if it doesn't exist)
%       var       - Variable name (string)
%       sz        - Initial size of the variable [rows, cols]
%       var_class - Data type string (e.g., 'double', 'single', 'int16')
%
%   Edited from 'savefast.m' by Timothy E. Holy
%   Copyright 2013 by Timothy E. Holy

    % Append .mat if necessary
    [filepath, filebase, ext] = fileparts(filename);
    if isempty(ext)
        filename = fullfile(filepath, [filebase '.mat']);
    end

    if exist(filename, 'file') == 0

        % Save a dummy variable, just to create the file
        dummy = 0; %#ok<NASGU>
        save(filename, '-v7.3', 'dummy', '-nocompression');

        % Delete the dummy
        fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
        H5L.delete(fid, 'dummy', 'H5P_DEFAULT');
        H5F.close(fid);

    end

    % Create the chunked dataset with unlimited max dimensions
    varname = ['/' var];
    chunk_rows = min(32000, max(1, sz(1)));
    chunk_cols = max(1, sz(2));
    h5create(filename, varname, [Inf, Inf], ...
        'DataType', var_class, ...
        'ChunkSize', [chunk_rows, chunk_cols], ...
        'Deflate', 0);

    % Write initial data as zeros to establish the initial extent
    % (h5create with Inf creates a 0x0 dataset; we need to set the extent)
    fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    dset_id = H5D.open(fid, varname);
    H5D.set_extent(dset_id, fliplr(sz)); % Set to initial size
    H5D.close(dset_id);
    H5F.close(fid);

end
