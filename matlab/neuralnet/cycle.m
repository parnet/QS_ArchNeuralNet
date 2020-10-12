function [dWA] = cycle(clas,NQGP_Input, WA, WB, dropindex, dropnumber)
NQGP_0 = WA'*NQGP_Input;
NQGP_1 = neuronLeakyReLU(NQGP_0);
NQGP_2 = dropout(NQGP_1,dropindex,dropnumber);
NQGP_3 = antidropout(NQGP_2,dropindex,64);
NQGP_4 = WB'*NQGP_3;
NQGP_5 = neuronSoftmaxMinTranslation(NQGP_4);
NQGP_6 = neuronSoftmax(NQGP_5);
result_nqgp = NQGP_6;
BNQGP_0 = outputErrorSignal(result_nqgp,target(clas)');
BNQGP_1 = neuronSoftmaxBackpass(NQGP_5,BNQGP_0);
BNQGP_2 = WB*BNQGP_1;
dWB = NQGP_3*BNQGP_1';
BNQGP_3 = restrictdropout(BNQGP_2,dropindex);
BNQGP_4 = neuronLeakyReLUBackpass(NQGP_2, BNQGP_3);
BNQGP_5 = antidropout(BNQGP_4, dropindex,64);
BNQGP_5'
dWA = NQGP_Input * BNQGP_5';
CNQGP = []
end
