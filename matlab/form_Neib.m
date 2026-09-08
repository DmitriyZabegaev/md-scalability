function Neib_list = form_Neib(Xi, Yi, Zi, Cell_list, ...
    Lx, Ly, Lz, r_cell_x, r_cell_y, r_cell_z, ...
    N_cell_x, N_cell_y, N_cell_z)
% FORM_NEIB Build the neighbor list for a particle.
%   The search includes the particle's cell and all directly neighboring
%   cells in a 3-by-3-by-3 cell neighborhood.

% Determine the cell containing the particle.
if (Xi > 0) && (Xi < Lx)
    nx = ceil(Xi / r_cell_x);
else
    if Xi < 0
        nx = 1;
    else
        nx = N_cell_x;
    end
end

if (Yi > 0) && (Yi < Ly)
    ny = ceil(Yi / r_cell_y);
else
    if Yi < 0
        ny = 1;
    else
        ny = N_cell_y;
    end
end

if (Zi > 0) && (Zi < Lz)
    nz = ceil(Zi / r_cell_z);
else
    if Zi < 0
        nz = 1;
    else
        nz = N_cell_z;
    end
end

% Collect particles from the current cell and its valid neighbors.
Neib_list = [];

for jx = -1:1
    for jy = -1:1
        for jz = -1:1
            if ((nx + jx) >= 1) && ((nx + jx) <= N_cell_x) && ...
               ((ny + jy) >= 1) && ((ny + jy) <= N_cell_y) && ...
               ((nz + jz) >= 1) && ((nz + jz) <= N_cell_z)

                Neib_list = [Neib_list, Cell_list{nx + jx, ny + jy, nz + jz}];
            end
        end
    end
end

end
