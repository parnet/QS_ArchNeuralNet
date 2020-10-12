#include "leakyreluneuron.h"

LeakyReLUNeuron::LeakyReLUNeuron()
{

}

LeakyReLUNeuron::~LeakyReLUNeuron()
{

}

number LeakyReLUNeuron::activate(size_t index, std::vector<number> &input){
    if(input[index] < 0.0){
        return slope*input[index];
    } else {
        return input[index];
    }
}

number LeakyReLUNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input){
    if(input[index] < 0.0){
        return errorsignal[index] * slope;
    }else{
        return errorsignal[index];
    }
}

number LeakyReLUNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input)
{
    if(input[inputindex] < 0.0){
        return errorsignal[errorindex] * slope;
    }else{
        return errorsignal[errorindex];
    }
}

number LeakyReLUNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output)
{
    if(input[inputindex] < 0.0){
        return errorsignal[errorindex] * slope;
    }else{
        return errorsignal[errorindex];
    }
}


number LeakyReLUNeuron::randomWeight(size_t numberOfNeurons){
    auto dist = std::normal_distribution<number>(0, sqrt(2.0/numberOfNeurons));
    return dist(RandomDevice::engine);
}
