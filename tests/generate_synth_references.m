% generate_synth_references.m
% Generate reference outputs from the MATLAB/Octave gridfit for comparison
% with the Python port. Exercises all interp x regularizer x solver combos
% on a deterministic synthetic trig surface.
%
% Requires: gridfit.m on the path (shipped in dev/gridfitdir/).
%
% Usage (from repo root):
%   octave --no-gui --path dev/gridfitdir tests/generate_synth_references.m

interps = {'triangle', 'bilinear', 'nearest'};
regularizers = {'gradient', 'diffusion', 'springs'};
% Octave lacks lsqr; use 'normal' (matches Python) and 'backslash' (cross-check)
solvers = {'normal', 'backslash'};

%% Build deterministic scattered data
% Trig surface: sin(4x+5y)*cos(7(x-y)) + exp(x+y) on [0,1]^2.
% Sample on a 26x26 meshgrid, keep every 3rd point (226 pts) so the data
% does not sit exactly on the fitting grid.
xi = linspace(0, 1, 26);
[xg, yg] = meshgrid(xi, xi);
x_all = xg(:);
y_all = yg(:);
z_all = sin(4*x_all + 5*y_all) .* cos(7*(x_all - y_all)) + exp(x_all + y_all);

idx = 1:3:length(x_all);
x_s = x_all(idx);
y_s = y_all(idx);
z_s = z_all(idx);

xnodes = linspace(0, 1, 21);
ynodes = linspace(0, 1, 21);

%% Run all parameter combinations
results = struct();

for ii = 1:length(interps)
  for ri = 1:length(regularizers)
    for si = 1:length(solvers)
      interp_name = interps{ii};
      reg_name = regularizers{ri};
      solver_name = solvers{si};

      key = [interp_name '_' reg_name '_' solver_name];
      fprintf('Running: %s\n', key);

      g = gridfit(x_s, y_s, z_s, xnodes, ynodes, ...
        'interp', interp_name, ...
        'regularizer', reg_name, ...
        'solver', solver_name, ...
        'smoothness', 1, ...
        'extend', 'never');

      results.(key) = g;
    end
  end
end

%% Anisotropic smoothness [1, 2]
aniso_combos = {
  'triangle', 'gradient', 'normal';
  'bilinear', 'diffusion', 'normal';
  'nearest',  'springs',   'normal';
};

for ci = 1:size(aniso_combos, 1)
  interp_name = aniso_combos{ci, 1};
  reg_name = aniso_combos{ci, 2};
  solver_name = aniso_combos{ci, 3};

  key = ['aniso_' interp_name '_' reg_name '_' solver_name];
  fprintf('Running: %s\n', key);

  g = gridfit(x_s, y_s, z_s, xnodes, ynodes, ...
    'interp', interp_name, ...
    'regularizer', reg_name, ...
    'solver', solver_name, ...
    'smoothness', [1, 2], ...
    'extend', 'never');

  results.(key) = g;
end

%% Save
synth_x = x_s;
synth_y = y_s;
synth_z = z_s;
synth_xnodes = xnodes;
synth_ynodes = ynodes;

outpath = fullfile(fileparts(mfilename('fullpath')), 'data', 'synth_references.mat');
save('-v7', outpath, 'results', ...
  'synth_x', 'synth_y', 'synth_z', 'synth_xnodes', 'synth_ynodes');
fprintf('Saved %s\n', outpath);
