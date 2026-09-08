function [X, Y, Z, Vx, Vy, Vz] = quick_Verle_par(X, Y, Z, Vx, Vy, Vz, N_steps, dt)
% QUICK_VERLE_PAR Run the parallel velocity-Verlet integration.
%   Particle accelerations are evaluated with finite differences of the
%   smoothed/truncated Lennard-Jones potential. MATLAB parfor is used to
%   parallelize the particle-wise acceleration calculations.
%
%   The original algorithm and numerical expressions are preserved.

global N_p Lx Ly Lz m1 C_units ...
    r_cell_x r_cell_y r_cell_z N_cell_x N_cell_y N_cell_z R_off

delta = 0.001;

%% Initial acceleration
Ax_old = zeros(1, N_p);
Ay_old = zeros(1, N_p);
Az_old = zeros(1, N_p);

Cell_list = form_cells(X, Y, Z, Lx, Ly, Lz, ...
    N_cell_x, r_cell_x, N_cell_y, r_cell_y, N_cell_z, r_cell_z);

parfor i = 1:N_p
    % Displacement vector used for numerical differentiation.
    D = zeros(1, N_p);
    D(i) = delta;

    Neib_list = form_Neib(X(i), Y(i), Z(i), Cell_list, ...
        Lx, Ly, Lz, r_cell_x, r_cell_y, r_cell_z, ...
        N_cell_x, N_cell_y, N_cell_z);

    Ui_dx = Ui(X + D, Y, Z, i, Neib_list, R_off);
    Ui_dy = Ui(X, Y + D, Z, i, Neib_list, R_off);
    Ui_dz = Ui(X, Y, Z + D, i, Neib_list, R_off);
    Ui_0 = Ui(X, Y, Z, i, Neib_list, R_off);

    Ax_old(i) = -(Ui_dx - Ui_0) / (C_units * m1 * delta);
    Ay_old(i) = -(Ui_dy - Ui_0) / (C_units * m1 * delta);
    Az_old(i) = -(Ui_dz - Ui_0) / (C_units * m1 * delta);
end

%% Velocity-Verlet integration
Ax = zeros(1, N_p);
Ay = zeros(1, N_p);
Az = zeros(1, N_p);

for j = 1:N_steps
    %% Reflect particles from the rigid walls
    parfor i = 1:N_p
        if X(i) > Lx
            Vx(i) = -abs(Vx(i));
        end
        if X(i) < 0
            Vx(i) = abs(Vx(i));
        end

        if Y(i) > Ly
            Vy(i) = -abs(Vy(i));
        end
        if Y(i) < 0
            Vy(i) = abs(Vy(i));
        end

        if Z(i) > Lz
            Vz(i) = -abs(Vz(i));
        end
        if Z(i) < 0
            Vz(i) = abs(Vz(i));
        end
    end

    %% Update particle coordinates
    X = X + Vx * dt + 0.5 * Ax_old * dt^2;
    Y = Y + Vy * dt + 0.5 * Ay_old * dt^2;
    Z = Z + Vz * dt + 0.5 * Az_old * dt^2;

    %% Rebuild the spatial cell list
    Cell_list = form_cells(X, Y, Z, Lx, Ly, Lz, ...
        N_cell_x, r_cell_x, N_cell_y, r_cell_y, N_cell_z, r_cell_z);

    %% Compute new accelerations
    parfor i = 1:N_p
        D = zeros(1, N_p);
        D(i) = delta;

        Neib_list = form_Neib(X(i), Y(i), Z(i), Cell_list, ...
            Lx, Ly, Lz, r_cell_x, r_cell_y, r_cell_z, ...
            N_cell_x, N_cell_y, N_cell_z);

        U_dx = Ui(X + D, Y, Z, i, Neib_list, R_off);
        U_dy = Ui(X, Y + D, Z, i, Neib_list, R_off);
        U_dz = Ui(X, Y, Z + D, i, Neib_list, R_off);
        U_0 = Ui(X, Y, Z, i, Neib_list, R_off);

        Ax(i) = -(U_dx - U_0) / (C_units * m1 * delta);
        Ay(i) = -(U_dy - U_0) / (C_units * m1 * delta);
        Az(i) = -(U_dz - U_0) / (C_units * m1 * delta);
    end

    %% Update velocities
    Vx = Vx + 0.5 * (Ax + Ax_old) * dt;
    Vy = Vy + 0.5 * (Ay + Ay_old) * dt;
    Vz = Vz + 0.5 * (Az + Az_old) * dt;

    %% Store the new accelerations for the next step
    Ax_old = Ax;
    Ay_old = Ay;
    Az_old = Az;
end

end
