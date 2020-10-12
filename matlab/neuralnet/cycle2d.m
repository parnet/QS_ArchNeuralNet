function [Kernel,Bias] =cycle(Input_0, Target_0, Kernel_0,Bias_0)

Input = Input_0;
Target = Target_0;
Kernel = Kernel_0;
Bias = Bias_0;
eta = 0.0005;
for i = 0:10000
    [Err,dB, dX,dK] = test2d(Input,Target, Kernel,Bias);
    Kernel = Kernel + eta*dK;
    
    Bias = Bias + eta*dB;
    disp(dB)
end
end