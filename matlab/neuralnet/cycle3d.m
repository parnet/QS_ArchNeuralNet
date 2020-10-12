function [Kernel,Bias] =cycle3d(Input_0, Target_0, Kernel_0,Bias_0,k)

Input = Input_0;
Target = Target_0;
Kernel = Kernel_0;
Bias = Bias_0;
eta = 0.001;
eta_bias = 0.002;



for i = 0:k
    [Err,dB, dX,dKC] = test3d(Input,Target, Kernel,Bias);
    %dKK = dKK*(1-alpha)+alpha*dKC; dKC = dKK;
    Kernel = Kernel + eta*dKC;
    Bias = Bias +eta_bias* dB;
    
    %vdw = beta1 * vdw + (1 - beta1) * dKC;
    %sdw = beta2 * sdw + (1 - beta2) * dKC.^2;
    %vdw_corrected = vdw / (1-beta1^(epoch+1));
    %sdw_corrected = sdw / (1-beta2^(epoch+1));
    %Kernel = Kernel +eta * (vdw_corrected ./ sqrt((sdw_corrected) + epsilon));
    
    disp(dB)
end
end