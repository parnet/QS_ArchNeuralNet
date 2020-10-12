#include "identityneuron.h"

IdentityNeuron::IdentityNeuron(){}

IdentityNeuron::~IdentityNeuron()
{

}

number IdentityNeuron::activate(size_t index, std::vector<number> &input){
    return input[index];
}

number IdentityNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input){
    return errorsignal[index];
}

number IdentityNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input){
    return errorsignal[errorindex];
}

number IdentityNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output){
    return errorsignal[errorindex];
}
