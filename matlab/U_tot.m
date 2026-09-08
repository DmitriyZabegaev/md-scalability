function out = U_tot(X, Y, Z)
% U_TOT Compute the total potential energy of the particle system.

R_cut = 2.5;

N = length(X);
s = 0;

for i = 1:N
    for j = 1:(i - 1)
        R = sqrt((X(i) - X(j))^2 + ...
                 (Y(i) - Y(j))^2 + ...
                 (Z(i) - Z(j))^2);

        if R < R_cut
            s = s + U_lj_cut(R);
        end
    end
end

out = s;

end
