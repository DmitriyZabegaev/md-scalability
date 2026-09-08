function s = Ui(X, Y, Z, i, Neib_list, R_off)
% UI Compute the potential-energy contribution of particle i.
%   Neib_list contains particles in the same or neighboring spatial cells.

N_neib = length(Neib_list);
s = 0;

for j = 1:N_neib
    k = Neib_list(j);

    r_ij = sqrt((X(i) - X(k))^2 + ...
                (Y(i) - Y(k))^2 + ...
                (Z(i) - Z(k))^2);

    if (r_ij < R_off) && (i ~= k)
        s = s + U_lj_cut(r_ij);
    end
end

end
