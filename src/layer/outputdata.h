#ifndef OUTPUTDATA_H
#define OUTPUTDATA_H

#include "environment.h"

class OutputData
{
public:
    std::vector<number> target;

    std::vector<number> input;
    std::vector<number> output;

    std::vector<number> rightErrorSignal;
    std::vector<size_t> activeErrorSignal;
    std::vector<number> leftErrorSignal;

public:
    OutputData();

    void setTarget(std::vector<number> target);
};

#endif // OUTPUTDATA_H
