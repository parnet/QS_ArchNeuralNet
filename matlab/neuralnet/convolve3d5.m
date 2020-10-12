function Conv = convolve3d5(Input,Kernel, Bias)
szInp = size(Input);
szSzInp = size(szInp);
if (szSzInp(2) == 4)
    Input = squeeze(Input(1,:,:,:));
end

Input = transformStorage(Input);

[m,n,p] = size(Input);
Result = convn(Input,Kernel);
Conv = Result(3:(m+2),3:(n+2),3:(p+2)) + Bias;
end
