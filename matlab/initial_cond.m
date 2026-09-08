function [X, Y, Z, Vx, Vy, Vz] = initial_cond(N, Lx, Ly, Lz, r_lim, V)
% INITIAL_COND Generate initial particle coordinates and velocities.
%   Particles are placed randomly in the simulation box while enforcing
%   a minimum initial separation r_lim. Initial velocity directions are
%   sampled from spherical coordinates with fixed speed V.

%% Initial particle coordinates
X = zeros(1, N);
Y = zeros(1, N);
Z = zeros(1, N);

X(1) = Lx * rand();
Y(1) = Ly * rand();
Z(1) = Lz * rand();

failed_to_place_N_particles = 0;

for i = 2:N
    X(i) = Lx * rand();
    Y(i) = Ly * rand();
    Z(i) = Lz * rand();

    % Check the distance from the new particle to all previously placed particles.
    k = 0;
    err = 1;

    while (err == 1) && (k <= 10)
        err = 0;

        for j = 1:(i - 1)
            R = sqrt((X(i) - X(j))^2 + ...
                     (Y(i) - Y(j))^2 + ...
                     (Z(i) - Z(j))^2);

            if R < r_lim
                err = 1;
                break;
            end
        end

        if err == 1
            X(i) = Lx * rand();
            Y(i) = Ly * rand();
            Z(i) = Lz * rand();
            k = k + 1;
        end
    end

    if k >= 10
        str = ['Failed to place ', num2str(N), ' particles. ', ...
               'Only ', num2str(i - 1), ' particles have been placed!']; %#ok<NASGU>
        failed_to_place_N_particles = 1;
        break;
    end
end

if failed_to_place_N_particles == 1
    X(i:N) = [];
    Y(i:N) = [];
    Z(i:N) = [];
end

N = length(X);

%% Initial particle velocities
Vx = zeros(1, N);
Vy = zeros(1, N);
Vz = zeros(1, N);

for i = 1:N
    phi = 2 * pi * rand();
    theta = pi * rand();

    Vx(i) = V * cos(phi) * sin(theta);
    Vy(i) = V * sin(phi) * sin(theta);
    Vz(i) = V * cos(theta);
end

end
