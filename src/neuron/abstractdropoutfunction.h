#ifndef ABSTRACTDROPOUTFUNCTION_H
#define ABSTRACTDROPOUTFUNCTION_H

#include "environment.h"

class AbstractDropoutFunction{
public:
    AbstractDropoutFunction();

    virtual ~AbstractDropoutFunction();

    virtual number apply() = 0;

    virtual number apply(size_t epoch) = 0;

    virtual number apply(size_t epoch, size_t layer) = 0;
};

#endif // ABSTRACTDROPOUTFUNCTION_H
