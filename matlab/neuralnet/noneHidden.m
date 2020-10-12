function dW = noneHidden(W, data, clas)
OutLayer_Input = W'*data;
OutLayer_Trans = neuronSoftmaxMinTranslation(OutLayer_Input);
OutLayer_Output = neuronSoftmax(OutLayer_Trans);

OutLayer_RightErrorSignal = outputErrorSignal(OutLayer_Output,target(clas)');
OutLayer_LeftErrorSignal = neuronSoftmaxBackpass(OutLayer_Trans,OutLayer_RightErrorSignal);
InLayer_RightErrorSignal = W*OutLayer_LeftErrorSignal;
dW = data*OutLayer_LeftErrorSignal';
end