function R = mask(A, val)
[m,n,p]= size(A);
R= zeros(m,n,p);
for i = 1:m
    for j = 1:n
        for k = 1:p
            if(A(i,j,k) > val)
                R(i,j,k) = 1;
            end
        end
    end
end
            