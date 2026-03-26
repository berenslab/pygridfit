% generate_synth_references.m
% Generate reference outputs from the MATLAB/Octave gridfit for comparison
% with the Python port. Exercises all interp x regularizer x solver combos
% on a deterministic synthetic surface with realistic scattered data.
%
% Requires: gridfit.m on the path (shipped in dev/gridfitdir/).
%
% Usage (from repo root):
%   octave --no-gui --path dev/gridfitdir tests/generate_synth_references.m

interps = {'triangle', 'bilinear', 'nearest'};
regularizers = {'gradient', 'diffusion', 'springs'};
% Octave lacks lsqr; use 'normal' (matches Python) and 'backslash' (cross-check)
solvers = {'normal', 'backslash'};

%% Build deterministic scattered data that mimics real-world properties:
%  - Irregularly spaced (not on a grid)
%  - Non-uniform density (clustered in some regions, sparse in others)
%  - Measurement noise
%  - Some points near grid boundaries
%  - Replicates (multiple z values at similar locations)
rand('seed', 42);

% Base: 200 irregularly scattered points via randn (clustered near center)
n1 = 200;
x1 = 0.5 + 0.25 * randn(n1, 1);
y1 = 0.5 + 0.25 * randn(n1, 1);

% Additional sparse points in corners (low-density regions)
n2 = 30;
corners_x = [0.05*ones(n2/3,1); 0.95*ones(n2/3,1); 0.05*rand(n2/3,1)];
corners_y = [0.05*rand(n2/3,1);  0.95*rand(n2/3,1); 0.95*ones(n2/3,1)];
x1 = [x1; corners_x];
y1 = [y1; corners_y];

% Clip to [0.01, 0.99] so extend='never' works with [0,1] grid
x1 = min(0.99, max(0.01, x1));
y1 = min(0.99, max(0.01, y1));

% True surface
z_true = sin(4*x1 + 5*y1) .* cos(7*(x1 - y1)) + exp(x1 + y1);

% Add measurement noise (small relative to signal range)
noise = 0.02 * randn(length(x1), 1);
z1 = z_true + noise;

% Add replicates: duplicate ~20 points with slightly different z
n_dup = 20;
idx_dup = 1:n_dup;
x1 = [x1; x1(idx_dup)];
y1 = [y1; y1(idx_dup)];
z1 = [z1; z1(idx_dup) + 0.01*randn(n_dup, 1)];

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

      g = gridfit(x1, y1, z1, xnodes, ynodes, ...
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

  g = gridfit(x1, y1, z1, xnodes, ynodes, ...
    'interp', interp_name, ...
    'regularizer', reg_name, ...
    'solver', solver_name, ...
    'smoothness', [1, 2], ...
    'extend', 'never');

  results.(key) = g;
end

%% Save
synth_x = x1;
synth_y = y1;
synth_z = z1;
synth_xnodes = xnodes;
synth_ynodes = ynodes;

outpath = fullfile(fileparts(mfilename('fullpath')), 'data', 'synth_references.mat');
save('-v7', outpath, 'results', ...
  'synth_x', 'synth_y', 'synth_z', 'synth_xnodes', 'synth_ynodes');
fprintf('Saved %s\n', outpath);
