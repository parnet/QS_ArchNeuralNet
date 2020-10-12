function I = KernelCover(Data)

I = zeros(24,24,24);
Kernel = zeros(5,5,5) +1;
for i = 1:28
    R = convn(Data(:,:,:,i),Kernel);
    I = I +R;
end
end