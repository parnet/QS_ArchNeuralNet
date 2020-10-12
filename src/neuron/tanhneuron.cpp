#include "tanhneuron.h"

TanhNeuron::TanhNeuron(){}

TanhNeuron::~TanhNeuron(){}

number TanhNeuron::activate(size_t index, std::vector<number> &input){
    return tanh(input[index]);
}

number TanhNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input){
    number t = tanh(input[index]);
    return (1.0 - t*t)*errorsignal[index];
}

number TanhNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input){
    number t = tanh(input[inputindex]);
    return (1.0 - t*t)*errorsignal[errorindex];
}

number TanhNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output){
    return (1.0 - output[outputindex]*output[outputindex])*errorsignal[errorindex];
}

number TanhNeuron::randomWeight(size_t numberOfNeurons){
    auto dist = std::normal_distribution<number>(0, sqrt(1.0/numberOfNeurons));
    return dist(RandomDevice::engine);
}
