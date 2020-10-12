function dX = inputErrorSignal2d(LES, Kernel)
[m,n] = size(LES);
dX = conv2(LES, reflect2d(Kernel));
dX = dX(2:(m+1),2:(n+1));
end