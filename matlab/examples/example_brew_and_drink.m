%% example_tea.m — End-to-end demo of the TEA class
%
% Demonstrates: create, read, append, discontinuity handling, refresh.

%% 1. Create a TEA file with regular continuous data
SR = 1000;
t = (0:4999)' / SR;
Samples = randn(5000, 3);

tea = TEA('demo.mat', SR, true, 't_units', 's');
tea.write(t, Samples);
disp(tea.info());

%% 2. Read by time range
[Data, t_out] = tea.read([1, 3], [1.0, 2.5], []);
fprintf('Read %d samples, channels [1,3], t=[%.3f, %.3f]\n', size(Data,1), t_out(1), t_out(end));

%% 3. Read by sample range
[Data2, t_out2] = tea.read([], [], [100, 500]);
fprintf('Read samples 100–500: %d samples\n', size(Data2, 1));

%% 4. Append more time samples
t2 = t(end) + (1:5000)' / SR;
s2 = randn(5000, 3);
tea.write(t2, s2);
fprintf('After append: N=%d, C=%d\n', tea.N, tea.C);

%% 5. Append channels
tea.write_channels(randn(10000, 2), [4, 5]);
fprintf('After channel append: N=%d, C=%d, ch_map=%s\n', tea.N, tea.C, mat2str(tea.ch_map));

%% 6. Discontinuous data
t_disc = [(0:999)'/SR; (3000:3999)'/SR];             % 2-second gap
s_disc = randn(2000, 1);
tea_disc = TEA('demo_disc.mat', SR, true, 't_units', 's');
tea_disc.write(t_disc, s_disc);

[~, ~, disc_info] = tea_disc.read([], [], []);
fprintf('Discontinuous: %d, gaps: %d\n', disc_info.is_discontinuous, size(disc_info.disc, 1));

%% 7. Irregular data
t_irr = sort(rand(200, 1) * 100);
tea_irr = TEA('demo_irreg.mat', [], false);
tea_irr.write(t_irr, randn(200, 2));
fprintf('Irregular: N=%d\n', tea_irr.N);

%% 8. Cleanup
delete('demo.mat', 'demo_disc.mat', 'demo_irreg.mat');
disp('Done!');
