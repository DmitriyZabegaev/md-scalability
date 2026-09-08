function MAIN()
% MAIN Run the molecular-dynamics simulation and performance experiment.
%   The program initializes the particle system, performs N_t integration
%   intervals using the parallel velocity-Verlet algorithm, records the
%   total energy, and measures the execution time with tic/toc.

%% Simulation parameters
global N_cell_x r_cell_x N_cell_y r_cell_y N_cell_z r_cell_z
global N_p Lx Ly Lz m1 C_units r_min R_off

N_proposed = 10000;

% Simulation box dimensions.
Lx = 200;
Ly = 200;
Lz = 200;

% Argon and unit-conversion parameters.
m1 = 39.938;
C_units = 9648.533977;

% Lennard-Jones length parameters.
r_lim = 3.822;
r_min = 3.822;
R_off = 2.5 * r_min;

% Initial particle speed.
V = 3;

%% Initial conditions
[X, Y, Z, Vx, Vy, Vz] = initial_cond(N_proposed, Lx, Ly, Lz, r_lim, V);
N_p = length(X);

%% Spatial cell decomposition
N_cell_x = floor(Lx / R_off);
r_cell_x = Lx / N_cell_x;

N_cell_y = floor(Ly / R_off);
r_cell_y = Ly / N_cell_y;

N_cell_z = floor(Lz / R_off);
r_cell_z = Lz / N_cell_z;

%% Output figures
h1 = figure;
h2 = figure;
h3 = figure;

%% Integration and timing parameters
dt = 0.01;
N_steps = 20;
N_t = 2;
E_total = NaN * zeros(1, N_t);

%% Main simulation loop
tic;

for i = 1:N_t
    [X, Y, Z, Vx, Vy, Vz] = quick_Verle_par(...
        X, Y, Z, Vx, Vy, Vz, N_steps, dt);

    E_total(1, i) = E_tot(X, Y, Z, Vx, Vy, Vz);

    fprintf('Iteration %d out of %d finished. Energy = %d\n', ...
        i, N_t, E_total(1, i));
end

toc;

%% Velocity distribution
N_bars = 100;
dV = 0.1;
bars = find_bars(Vx, Vy, Vz, N_bars, dV);

%% Particle positions
figure(h1);
plot3(X, Y, Z, 'ob');
axis([0 Lx 0 Ly 0 Lz]);
title('Particle positions');
xlabel('X');
ylabel('Y');
zlabel('Z');

%% Total energy
figure(h2);
plot(1:N_t, E_total, '-b');
title('Total energy');
xlabel('Iteration');
ylabel('Energy');

%% Velocity distribution
figure(h3);
bar(bars);
title('Particle speed distribution');
xlabel('Velocity bin');
ylabel('Number of particles');

end
