function R = sumOut(A)
R = zeros(20,20,20,1);
for i = 1:28
    R(:,:,:,1) = R(:,:,:,1) + A(:,:,:,i);
end
end