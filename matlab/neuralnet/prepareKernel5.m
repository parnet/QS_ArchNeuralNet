function Kernel = prepareKernel5(raw)
W = reshape(reshape(raw,5*5*5,1),5,5,5);
Kernel = zeros(5,5,5);
for i = 1:5
    for j = 1:5
        for k = 1:5
            Kernel(i,j,k) = W(j,i,k);
        end
    end
end

%Kernel = reflect3d(Kernel);
end