#ifndef FULLYCONNECTEDDRIVER_H
#define FULLYCONNECTEDDRIVER_H

#include "abstractlayer.h"
#include "fullyconnecteddescription.h"

#include <weightadamoptimization.h>
#include <weightgradientdescent.h>

class FullyConnectedDriver{
public:
    typedef  WeightAdamOptimization Updater;
public:

    size_t leftSize;
    size_t rightSize;
    
    size_t szLeftActive;
    size_t szRightActive;

    size_t szBatchLeftActive;
    size_t szBatchRightActive;
    
    Updater updater;
    std::vector<std::vector<Updater::Variable>> connections;
public:
    FullyConnectedDriver();

    FullyConnectedDriver(size_t szPrevNeuron, size_t szNextNeuron);

    void displayWeights();

    void displayWeightChanges();

    void update(size_t epoch);

    void serialize(std::ostream & out);

    // todo
    void reset();

    void apply(FullyConnectedDescription desc, AbstractLayer *prev);

    void change(FullyConnectedDescription desc, AbstractLayer *prev);

};

#endif // FULLYCONNECTEDDRIVER_H
