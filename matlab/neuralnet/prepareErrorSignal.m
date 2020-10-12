function Error = prepareErrorSignal(raw)
W = reshape(reshape(raw,10*10*10,1),10,10,10);
Kernel = zeros(10,10,10);
for i = 1:10
    for j = 1:10
        for k = 1:10
            Kernel(i,j,k) = W(j,i,k);
        end
    end
end
Error = Kernel;
%Kernel = reflect3d(Kernel);
end