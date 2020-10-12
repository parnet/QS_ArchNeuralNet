function Conv = convolve3d(Input,Kernel, Bias)
szInp = size(Input);
szSzInp = size(szInp);
if (szSzInp(2) == 4)
    Input = squeeze(Input(1,:,:,:));
end

[m,n,p] = size(Input);
Result = convn(Input,Kernel);
Conv = Result(2:(m+1),2:(n+1),2:(p+1)) + Bias;
end
