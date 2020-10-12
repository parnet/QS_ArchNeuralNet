function Kernel = prepareKernel(raw)
W = reshape(reshape(raw,27,1),3,3,3);
Kernel = zeros(3,3,3);
for i = 1:3
    for j = 1:3
        for k = 1:3
            Kernel(i,j,k) = W(j,i,k);
        end
    end
end

%Kernel = reflect3d(Kernel);
end