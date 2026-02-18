%% Example: Brew and Drink TEA files
% Demonstrates creating, reading, appending, and inspecting TEA files.

%% Setup
% Add the TEA matlab directory to the path
tea_dir = fileparts(fileparts(mfilename('fullpath'))); % go up from examples/ to matlab/
addpath(tea_dir);

% Temporary file for this example
temp_file = fullfile(tempdir, 'tea_example.mat');
if exist(temp_file, 'file'), delete(temp_file); end

%% 1. Create a continuous TEA file
fprintf('=== 1. Create a continuous TEA file ===\n');

SR = 1000; % 1000 Hz
duration = 5; % 5 seconds
N = SR * duration;
t = (0:N-1)' / SR; % time in seconds
ch1 = sin(2*pi*10*t); % 10 Hz sine
ch2 = cos(2*pi*25*t); % 25 Hz cosine
Samples = [ch1, ch2];

brew_TEA(temp_file, t, Samples, SR, true, ...
    't_units', 's', ...
    'ch_map', [1, 2]);

%% 2. Read it back
fprintf('\n=== 2. Read it back ===\n');

[Data, t_out, disc_info] = drink_TEA(temp_file, [], [], []);
fprintf('Loaded %d samples, %d channels\n', size(Data, 1), size(Data, 2));
fprintf('Discontinuous? %d\n', disc_info.is_discontinuous);

%% 3. Read a time range
fprintf('\n=== 3. Read a time range ===\n');

[Data_seg, t_seg] = drink_TEA(temp_file, [1], [1.0, 2.0], []);
fprintf('Segment: %d samples of channel 1, t=[%.3f, %.3f]\n', ...
    length(t_seg), t_seg(1), t_seg(end));

%% 4. Read a sample range
fprintf('\n=== 4. Read a sample range ===\n');

[Data_sr, t_sr] = drink_TEA(temp_file, [2], [], [100, 200]);
fprintf('Sample range: %d samples of channel 2\n', size(Data_sr, 1));

%% 5. Append in time
fprintf('\n=== 5. Append more data in time ===\n');

t_new = t(end) + (1:N)' / SR; % next 5 seconds
ch1_new = sin(2*pi*10*t_new);
ch2_new = cos(2*pi*25*t_new);
Samples_new = [ch1_new, ch2_new];

brew_TEA(temp_file, t_new, Samples_new, SR, true, 'mode', 'append_time');

[Data_full, t_full, disc_full] = drink_TEA(temp_file, [], [], []);
fprintf('After append: %d samples total\n', size(Data_full, 1));
fprintf('Discontinuous? %d\n', disc_full.is_discontinuous);

%% 6. Create a discontinuous TEA file
fprintf('\n=== 6. Create a discontinuous file ===\n');

temp_disc = fullfile(tempdir, 'tea_disc_example.mat');
if exist(temp_disc, 'file'), delete(temp_disc); end

% Two segments separated by a 2-second gap
t1 = (0:999)' / SR; % 0 to 0.999 s
t2 = (3000:3999)' / SR; % 3.0 to 3.999 s (gap from 1.0 to 3.0)
t_disc = [t1; t2];
s_disc = randn(length(t_disc), 1);

brew_TEA(temp_disc, t_disc, s_disc, SR, true, 't_units', 's');

[~, ~, disc_info2] = drink_TEA(temp_disc, [], [], []);
fprintf('Discontinuous? %d\n', disc_info2.is_discontinuous);
fprintf('Continuous blocks:\n');
disp(disc_info2.cont);
fprintf('Discontinuities:\n');
disp(disc_info2.disc);

%% 7. Demonstrate refresh_TEA
fprintf('\n=== 7. Refresh a file missing dependents ===\n');

temp_refresh = fullfile(tempdir, 'tea_refresh_example.mat');
if exist(temp_refresh, 'file'), delete(temp_refresh); end

% Manually create a minimal TEA file without dependents
t_r = (0:4999)' / 500;
s_r = randn(5000, 2);
save(temp_refresh, 't_r', '-v7.3');
mf = matfile(temp_refresh, 'Writable', true);
% Rename to proper TEA variables
mf.t = t_r;
mf.Samples = s_r;
mf.SR = 500;
mf.isRegular = true;

fprintf('Before refresh:\n');
vars_before = whos(mf);
fprintf('  Variables: %s\n', strjoin({vars_before.name}, ', '));

refresh_TEA(temp_refresh);

mf2 = matfile(temp_refresh);
vars_after = whos(mf2);
fprintf('After refresh:\n');
fprintf('  Variables: %s\n', strjoin({vars_after.name}, ', '));

%% 8. Plot
fprintf('\n=== 8. Plot example ===\n');

figure('Name', 'TEA Example');

subplot(2,1,1);
plot(t_full, Data_full);
xlabel('Time (s)'); ylabel('Amplitude');
title('Continuous TEA file (after append)');
legend('ch1: 10 Hz sine', 'ch2: 25 Hz cosine');

subplot(2,1,2);
[Data_disc_plot, t_disc_plot] = drink_TEA(temp_disc, [], [], []);
plot(t_disc_plot, Data_disc_plot, '.-');
xlabel('Time (s)'); ylabel('Amplitude');
title('Discontinuous TEA file (2s gap)');

%% Cleanup
delete(temp_file);
delete(temp_disc);
delete(temp_refresh);
fprintf('\nExample complete. Temp files cleaned up.\n');
