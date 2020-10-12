function RKernel = reflect2d(Kernel)

RKernel = flip(flip(Kernel)')';

end