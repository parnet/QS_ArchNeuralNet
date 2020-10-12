#include "softmaxneuron.h"

SoftmaxNeuron::SoftmaxNeuron(){}

SoftmaxNeuron::~SoftmaxNeuron(){}

number SoftmaxNeuron::activate(size_t index, std::vector<number> &input)
{
    number translation = std::numeric_limits<number>::min();
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

    if(isinf(sum)){
        sDebug() << "Warning sum is inf";
    }

    return exp(input[index]) / sum;
}

number SoftmaxNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input)
{
    size_t szActive = errorsignal.size();
    number sum = 0;
    for (size_t a = 0; a < szActive; ++a) {
        number sumval = input[a];
        sum += exp(sumval);
    }


    number delta_nk;
    number activation_functionSoftMax_n = exp(input[index])/sum;
    number leftErrorSignal = 0.0;

    for (size_t a = 0; a < szActive; ++a) {
        number difference = errorsignal[a];

        if (a == index) {
            delta_nk = 1;
        } else {
            delta_nk = 0;
        }
        number activation_functionSoftMax_k = exp(input[a])/sum;
        leftErrorSignal += difference * activation_functionSoftMax_n * (delta_nk - activation_functionSoftMax_k);
    }
    return leftErrorSignal;
}

number SoftmaxNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input)
{
    size_t szActive = errorsignal.size();
    number delta_nk;
    number sum = 0;
    for (size_t a = 0; a < szActive; ++a) {
        number sumval = input[a];
        sum += exp(sumval);
    }

    number activation_functionSoftMax_n = exp(input[inputindex]) / sum;
    number leftErrorSignal = 0.0;

    for (size_t a = 0; a < szActive; ++a) {
        number difference = errorsignal[a];

        if (a == errorindex) {
            delta_nk = 1;
        } else {
            delta_nk = 0;
        }
        number activation_functionSoftMax_k = exp(input[a]) / sum;
        leftErrorSignal += difference * activation_functionSoftMax_n * (delta_nk - activation_functionSoftMax_k);
    }
    return leftErrorSignal;
}

number SoftmaxNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output)
{
    size_t szActive = errorsignal.size();
    number delta_nk;
    number activation_functionSoftMax_n = output[outputindex];
    number leftErrorSignal = 0.0;

    for (size_t a = 0; a < szActive; ++a) {
        number difference = errorsignal[a];

        if (a == errorindex) {
            delta_nk = 1;
        } else {
            delta_nk = 0;
        }
        number activation_functionSoftMax_k = output[a];
        leftErrorSignal += difference * activation_functionSoftMax_n * (delta_nk - activation_functionSoftMax_k);
    }
    return leftErrorSignal;
}
