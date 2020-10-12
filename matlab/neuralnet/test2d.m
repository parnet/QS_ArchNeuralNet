function [Err,dB, dX,dK] = test2d(X,T, K, B)
Conv = convolve2d(X,K, B);
Act = activate2d(Conv);
[Output,Trace] = pooling2d(Act);

Err = calcError2d(T,Output);
LES = leftErrorSignal2d(Err, Conv, Trace);

dB = biasChange2d(LES);
dX = inputErrorSignal2d(LES, K);
dK = kernelChange2d(LES,X);
end