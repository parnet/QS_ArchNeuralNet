#include "reluneuron.h"

ReLUNeuron::ReLUNeuron()
{

}

ReLUNeuron::~ReLUNeuron()
{

}

number ReLUNeuron::activate(size_t index, std::vector<number> &input)
{
    if(input[index] < 0.0){
        return 0;
    } else {
        return input[index];
    }
}

number ReLUNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input)
{
    if(input[index] < 0.0){
        return 0.0;
    }else{
        return errorsignal[index];
    }
}

number ReLUNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input)
{
    if(input[inputindex] < 0.0){
        return 0.0;
    }else{
        return errorsignal[errorindex];
    }
}

number ReLUNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output)
{
    if(input[inputindex] < 0.0){
        return 0.0;
    }else{
        return errorsignal[errorindex];
    }
}
