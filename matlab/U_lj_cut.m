function out = U_lj_cut(r)
% U_LJ_CUT Compute the smoothed and truncated Lennard-Jones potential.
%   The potential is fully active below R_on, smoothly switched off
%   between R_on and R_off, and zero beyond R_off.

% Lennard-Jones parameters and cutoff radii.
epsilon = 0.01;
r_min = 3.822;
R_on = 2 * r_min;
R_off = 2.5 * r_min;

N = length(r);
w = zeros(1, N);

for i = 1:N
    if r(i) < R_on
        w(i) = 1;
    end

    if (r(i) >= R_on) && (r(i) <= R_off)
        w(i) = (R_off^2 - r(i)^2)^2 * ...
            (R_off^2 - 3 * R_on^2 + 2 * r(i)^2) / ...
            (R_off^2 - R_on^2)^3;
    end

    % Preserve the original implementation exactly.
    if r > R_off
        w(i) = 0;
    end
end

out = w .* epsilon .* ((r_min ./ r).^12 - 2 * (r_min ./ r).^6);

end
