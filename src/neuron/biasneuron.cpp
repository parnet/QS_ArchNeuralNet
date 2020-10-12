#include "biasneuron.h"

BiasNeuron::BiasNeuron(){}

BiasNeuron::~BiasNeuron(){}

number BiasNeuron::activate(size_t index, std::vector<number> &input) {
   return 1.0;
}

number BiasNeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) {
    sDebug() << "Bias Neuron Backprop requested";
    return 0.0;
}

number BiasNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input){
    sDebug() << "Bias Neuron Backprop requested";
    return 0.0;
}

number BiasNeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output){
    sDebug() << "Bias Neuron Backprop requested";
    return 0.0;
}

bool BiasNeuron::dropout(number probability) {
    return true;
}
