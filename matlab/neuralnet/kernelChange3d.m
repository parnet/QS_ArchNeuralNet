function dW = kernelChange3d(leftErrorSignal,Input)
modLES = reflect3d(leftErrorSignal);
%modLES = leftErrorSignal;
dS = convn(Input,modLES);
[m,n,p] = size(dS);

midm = floor(m/2)+1;
midn = floor(n/2)+1;
midp = floor(p/2)+1;

dW = dS(midm-1:midm+1,midn-1:midn+1,midp-1:midp+1);
dW = reflect3d(dW);
end