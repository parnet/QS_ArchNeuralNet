function dX = inputErrorSignal3d(LES, Kernel)
    [m,n,p] = size(LES);
    dX = convn((LES), reflect3d(Kernel));
    dX = dX(2:(m+1),2:(n+1),2:(p+1));
end

    %[m,n,p] = size(LES);
    %dX = convn(LES, reflect3d(Kernel));
    %dX = dX(2:(m+1),2:(n+1),2:(p+1));