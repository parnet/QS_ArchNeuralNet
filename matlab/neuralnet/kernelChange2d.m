function dW = kernelChange2d(leftErrorSignal,Input)
modLES = flip(flip(leftErrorSignal)')';
%modLES = leftErrorSignal;
dS = conv2(Input,modLES);
[m,n] = size(dS);
midm = floor(m/2)+1;
midn = floor(n/2)+1;
dW = dS(midm-1:midm+1,midn-1:midn+1);
dW = flip(flip(dW')');
end