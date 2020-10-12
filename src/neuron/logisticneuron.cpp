#include "logisticneuron.h"

LogisticNeuron::LogisticNeuron()
{

}

LogisticNeuron::~LogisticNeuron()
{

}

number LogisticNeuron::activate(size_t index, std::vector<number> &input)
{
    return 1.0 / (1 + exp(-(input[index])));
}

number LogisticNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input)
{
    double output = 1.0 / (1 + exp(-(input[index])));
    return errorsignal[index]* output* (1 - output);
}

number LogisticNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input){
    double output = 1.0 / (1 + exp(-(input[inputindex])));
    return errorsignal[errorindex]* output* (1 - output);
}

number LogisticNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output){
    return errorsignal[errorindex]* output[outputindex]* (1 - output[outputindex]);
}
