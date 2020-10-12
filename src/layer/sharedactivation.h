#ifndef SHAREDACTIVATION_H
#define SHAREDACTIVATION_H

#include "environment.h"

class SharedActivation {
public:
    std::vector<size_t> active;
    size_t fullsize;
public:
    SharedActivation();
};

#endif // SHAREDACTIVATION_H
