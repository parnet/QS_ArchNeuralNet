function Conv = convolve2d(Input,Kernel, Bias)
[m,n] = size(Input);
Result = conv2(Input,Kernel);
Conv = Result(2:(m+1),2:(n+1)) + Bias;
end
