function [Err,dB, dX,dK] = test3d(X,T, K, B)
[Conv, Output,Trace] = forward3d(X,K,B);

Err = calcError3d(T,Output);
LES = leftErrorSignal3d(Err, Conv, Trace);

dB = biasChange3d(LES);
dX = inputErrorSignal3d(LES, K);
dK = kernelChange3d(LES,X);
end