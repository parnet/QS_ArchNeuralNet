#ifndef NORMALIZATIONDATA_H
#define NORMALIZATIONDATA_H

#include "environment.h"

class NormalizationData
{
public:

    std::vector<number> input;

    std::vector<number> normalized; // dimension {batchsize} multiplied by {size of active neurons}

    std::vector<number> output; // dimension {batchsize} multiplied by {size of active neurons}

    std::vector<number> rightErrorSignal;

    std::vector<number> leftErrorSignal;

public:
    NormalizationData();


};

#endif // NORMALIZATIONDATA_H
