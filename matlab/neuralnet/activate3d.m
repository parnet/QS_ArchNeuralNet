function Active = activate3d(Conv)
slp = 0.01;
[m,n,p]= size(Conv);
Active = zeros(m,n,p);
for i = 1:m
    for j = 1:n
        for k = 1:p
        if(Conv(i,j,k) < 0)
            Active(i,j,k) = slp * Conv(i,j,k);
        else
            Active(i,j,k) = Conv(i,j,k);
        end
        end
    end
end
