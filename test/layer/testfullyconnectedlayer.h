#ifndef TESTFULLYCONNECTEDLAYER_H
#define TESTFULLYCONNECTEDLAYER_H

#include "environment.h"

#include <fullyconnectedlayer.h>

class __TestFullyConnectedLayer
{
private:
    static bool checkGradients(std::vector<std::vector<number>> expected, FullyConnectedLayer *layer);

public:
    __TestFullyConnectedLayer() = delete;

    static int all();

    static bool cycle_dd_dd();

    static bool cycle_ds_dd();

    static bool cycle_ds_sd();

    static bool cycle_dd_sd();




};

#endif // TESTFULLYCONNECTEDLAYER_H
