function Active = activate2d(Conv)
slp = 0.01;
[m,n]= size(Conv);
Active = zeros(m,n);
for i = 1:m
    for j = 1:n
        if(Conv(i,j) < 0)
            Active(i,j) = slp * Conv(i,j);
        else
            Active(i,j) = Conv(i,j);
        end
    end
end
