function out = find_bars(Vx, Vy, Vz, N_bars, dV)
% FIND_BARS Build a histogram of particle speeds.
%   N_bars specifies the number of velocity intervals and dV specifies
%   the width of each interval.

V = sqrt(Vx.^2 + Vy.^2 + Vz.^2);
V = sort(V);
bars = zeros(1, N_bars);

for i = 1:N_bars
    s = 0;

    for j = 1:length(V)
        if (V(j) > (i - 1) * dV) && (V(j) < i * dV)
            s = s + 1;
        end
    end

    bars(i) = s;
end

out = bars;

end
