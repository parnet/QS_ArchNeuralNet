function [Err] = calcError3d(target,output)
[m,n,p] = size(target);

Err = zeros(m,n,p);

for i = 1:m 
    for j = 1:n
        for k = 1:p
        Err(i,j,k) = target(i,j,k) - output(i,j,k);
        end
    end
end
end
