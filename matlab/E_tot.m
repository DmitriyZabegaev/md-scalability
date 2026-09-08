function out = E_tot(X, Y, Z, Vx, Vy, Vz)
% E_TOT Compute the total energy of the particle system.
%   The total energy is the sum of potential and kinetic energies.
%   The original cutoff and kinetic-energy expression are preserved.

m1 = 39.938;
C_units = 9648.533977;
R_cut = 2.5;

N = length(X);
s1 = 0;

for i = 1:N
    for j = 1:(i - 1)
        R = sqrt((X(i) - X(j))^2 + ...
                 (Y(i) - Y(j))^2 + ...
                 (Z(i) - Z(j))^2);

        if R < R_cut
            s1 = s1 + U_lj_cut(R);
        end
    end
end

s2 = 0;
for i = 1:N
    s2 = s2 + C_units * m1 * ...
        (Vx(i)^2 + Vy(i)^1 + Vz(i)^2) / 2;
end

out = s1 + s2;

end
