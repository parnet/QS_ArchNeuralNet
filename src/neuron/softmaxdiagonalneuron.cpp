#include "softmaxdiagonalneuron.h"

SoftmaxDiagonalNeuron::SoftmaxDiagonalNeuron()
{

}

SoftmaxDiagonalNeuron::~SoftmaxDiagonalNeuron()
{

}

number SoftmaxDiagonalNeuron::activate(size_t index, std::vector<number> &input)
{
    number translation = std::numeric_limits<number>::max();
    size_t szActive = input.size();
    for (size_t a = 0; a < szActive; ++a) {
        if (input[a] < translation){
            translation = input[a];
        }
    }
    number sum = 0;
    for (size_t a = 0; a < szActive; ++a) {
        number sumval = input[a] - translation;
        input[a] = sumval;
        sum += exp(sumval);
    }
    return exp(input[index]) / sum;
}

number SoftmaxDiagonalNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input){
    number activation_functionSoftMax_n = activate(index,input);
    return errorsignal[index] * activation_functionSoftMax_n * (1 - activation_functionSoftMax_n);
}

number SoftmaxDiagonalNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input)
{
    number activation_functionSoftMax_n = activate(inputindex,input);
    return errorsignal[errorindex] * activation_functionSoftMax_n * (1 - activation_functionSoftMax_n);
}

number SoftmaxDiagonalNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output)
{
    return errorsignal[errorindex] * output[outputindex] * (1 - output[outputindex]);
}
