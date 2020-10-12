#include "abstractneuron.h"

AbstractNeuron::AbstractNeuron(){}

AbstractNeuron::~AbstractNeuron(){}

number AbstractNeuron::randomWeight(size_t numberOfNeurons){
    auto dist = std::normal_distribution<number>(0.0, 1.0); // todo uniform distribution?
    return dist(RandomDevice::engine);
}

number AbstractNeuron::randomWeight(size_t rows, size_t columns){
    auto dist = std::normal_distribution<number>(0, sqrt(6.0/(rows+columns)));
    return dist(RandomDevice::engine);
}

bool AbstractNeuron::dropout(number probability){
    std::bernoulli_distribution dist(1.0 - probability);
    if (dist(RandomDevice::engine)) {
        return true;
    } else {
        return false;
    }
}

void AbstractNeuron::update(size_t epoch){}

number AbstractNeuron::updater(size_t epoch ){
    return 0.0;
}
