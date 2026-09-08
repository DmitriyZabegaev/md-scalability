function Cell_list = form_cells(X, Y, Z, Lx, Ly, Lz, ...
    N_cell_x, r_cell_x, N_cell_y, r_cell_y, N_cell_z, r_cell_z)
% FORM_CELLS Assign particles to the spatial cell list.
%   Each cell stores the indices of particles located in that cell.
%   The cell structure is used to restrict short-range neighbor searches.

Cell_list = cell(N_cell_x, N_cell_y, N_cell_z);
N_p = length(X);

for i = 1:N_p
    % Determine the cell index along the x-axis.
    if (X(i) > 0) && (X(i) < Lx)
        nx = ceil(X(i) / r_cell_x);
    else
        if X(i) < 0
            nx = 1;
        else
            nx = N_cell_x;
        end
    end

    % Determine the cell index along the y-axis.
    if (Y(i) > 0) && (Y(i) < Ly)
        ny = ceil(Y(i) / r_cell_y);
    else
        if Y(i) < 0
            ny = 1;
        else
            ny = N_cell_y;
        end
    end

    % Determine the cell index along the z-axis.
    if (Z(i) > 0) && (Z(i) < Lz)
        nz = ceil(Z(i) / r_cell_z);
    else
        if Z(i) < 0
            nz = 1;
        else
            nz = N_cell_z;
        end
    end

    % Append the particle index to the selected cell.
    Cell_list(nx, ny, nz) = {[Cell_list{nx, ny, nz}, i]};
end

end
