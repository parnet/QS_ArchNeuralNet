#ifndef GENERALDATA_H
#define GENERALDATA_H
#include "environment.h"

class GeneralData {
public:
    std::vector<number> output;
    std::vector<size_t> activeOutput;

    std::vector<number> errorSignal;
    std::vector<size_t> activeErrorSignal;
public:
    GeneralData();
};

#endif // GENERALDATA_H
