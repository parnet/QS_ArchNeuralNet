function [Conv, Output,Trace] = forward3d(X,K,B)
Conv = convolve3d(X,K, B);
Act = activate3d(Conv);
[Output,Trace] = pooling3d(Act);
end