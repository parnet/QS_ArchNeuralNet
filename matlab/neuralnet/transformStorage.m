function [B] = transformStorage(A)
[m,n,p] = size(A);
B = zeros(p,n,m);

for i = 1:m
    for j = 1:m
        for k = 1:p
            B(k,j,i) = A(i,j,k);
        end
    end
end

