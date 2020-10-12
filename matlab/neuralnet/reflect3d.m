function RKernel = reflect3d(Kernel)
[m,n,p] = size(Kernel);
RKernel = zeros(m,n,p);
for i = 1 : m
    for j = 1:n
        for k = 1:p
            RKernel(i,j,k) = Kernel(m-i+1,n-j+1,p-k+1);
        end
    end
end
end