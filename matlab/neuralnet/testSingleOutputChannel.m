function [ConvT] = testSingleOutputChannel(Data,Kernel_1_1,Kernel_1_2,Kernel_1_3,Kernel_1_4,Kernel_1_5,Kernel_1_6,Kernel_1_7,Kernel_1_8,Kernel_1_9,Kernel_1_10, Kernel_1_11,Kernel_1_12,Kernel_1_13,Kernel_1_14,Kernel_1_15,Kernel_1_16,Kernel_1_17,Kernel_1_18,Kernel_1_19,Kernel_1_20,Kernel_1_21,Kernel_1_22,Kernel_1_23,Kernel_1_24,Kernel_1_25,Kernel_1_26,Kernel_1_27,Kernel_1_28)

W_01 = prepareKernel5(Kernel_1_1);
W_02 = prepareKernel5(Kernel_1_2);
W_03 = prepareKernel5(Kernel_1_3);
W_04 = prepareKernel5(Kernel_1_4);
W_05 = prepareKernel5(Kernel_1_5);
W_06 = prepareKernel5(Kernel_1_6);
W_07 = prepareKernel5(Kernel_1_7);
W_08 = prepareKernel5(Kernel_1_8);
W_09 = prepareKernel5(Kernel_1_9);
W_10 = prepareKernel5(Kernel_1_10);
W_11 = prepareKernel5(Kernel_1_11);
W_12 = prepareKernel5(Kernel_1_12);
W_13 = prepareKernel5(Kernel_1_13);
W_14 = prepareKernel5(Kernel_1_14);
W_15 = prepareKernel5(Kernel_1_15);
W_16 = prepareKernel5(Kernel_1_16);
W_17 = prepareKernel5(Kernel_1_17);
W_18 = prepareKernel5(Kernel_1_18);
W_19 = prepareKernel5(Kernel_1_19);
W_20 = prepareKernel5(Kernel_1_20);
W_21 = prepareKernel5(Kernel_1_21);
W_22 = prepareKernel5(Kernel_1_22);
W_23 = prepareKernel5(Kernel_1_23);
W_24 = prepareKernel5(Kernel_1_24);
W_25 = prepareKernel5(Kernel_1_25);
W_26 = prepareKernel5(Kernel_1_26);
W_27 = prepareKernel5(Kernel_1_27);
W_28 = prepareKernel5(Kernel_1_28);

Bias = 0;
Conv_01 = convolve3d5(Data(:,:,:,1), W_01,Bias);
Conv_02 = convolve3d5(Data(:,:,:,2), W_02,Bias);
Conv_03 = convolve3d5(Data(:,:,:,3), W_03,Bias);
Conv_04 = convolve3d5(Data(:,:,:,4), W_04,Bias);
Conv_05 = convolve3d5(Data(:,:,:,5), W_05,Bias);
Conv_06 = convolve3d5(Data(:,:,:,6), W_06,Bias);
Conv_07 = convolve3d5(Data(:,:,:,7), W_07,Bias);
Conv_08 = convolve3d5(Data(:,:,:,8), W_08,Bias);
Conv_09 = convolve3d5(Data(:,:,:,9), W_09,Bias);
Conv_10 = convolve3d5(Data(:,:,:,10), W_10,Bias);
Conv_11 = convolve3d5(Data(:,:,:,11), W_11,Bias);
Conv_12 = convolve3d5(Data(:,:,:,12), W_12,Bias);
Conv_13 = convolve3d5(Data(:,:,:,13), W_13,Bias);
Conv_14 = convolve3d5(Data(:,:,:,14), W_14,Bias);
Conv_15 = convolve3d5(Data(:,:,:,15), W_15,Bias);
Conv_16 = convolve3d5(Data(:,:,:,16), W_16,Bias);
Conv_17 = convolve3d5(Data(:,:,:,17), W_17,Bias);
Conv_18 = convolve3d5(Data(:,:,:,18), W_18,Bias);
Conv_19 = convolve3d5(Data(:,:,:,19), W_19,Bias);
Conv_20 = convolve3d5(Data(:,:,:,20), W_20,Bias);
Conv_21 = convolve3d5(Data(:,:,:,21), W_21,Bias);
Conv_22 = convolve3d5(Data(:,:,:,22), W_22,Bias);
Conv_23 = convolve3d5(Data(:,:,:,23), W_23,Bias);
Conv_24 = convolve3d5(Data(:,:,:,24), W_24,Bias);
Conv_25 = convolve3d5(Data(:,:,:,25), W_25,Bias);
Conv_26 = convolve3d5(Data(:,:,:,26), W_26,Bias);
Conv_27 = convolve3d5(Data(:,:,:,27), W_27,Bias);
Conv_28 = convolve3d5(Data(:,:,:,28), W_28,Bias);



ConvT = Conv_01+Conv_02+Conv_03+Conv_04+Conv_05+Conv_06+Conv_07+Conv_08+Conv_09+Conv_10+Conv_11+Conv_12+Conv_13+Conv_14+Conv_15+Conv_16+Conv_17+Conv_18+Conv_19+Conv_20+Conv_21+Conv_22+Conv_23+Conv_24+Conv_25+Conv_26+Conv_27+Conv_28;

end