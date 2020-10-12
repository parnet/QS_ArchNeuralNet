function [Err] = calcError2d(target,output)
[m,n] = size(target);

Err = zeros(m,n);

for i = 1:m 
    for j = 1:n
        Err(i,j) = target(i,j) - output(i,j);
    end
end
end
