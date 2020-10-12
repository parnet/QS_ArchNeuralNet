#ifndef ABSTRACTNEURON_H
#define ABSTRACTNEURON_H
#include "environment.h"
#include "randomdevice.h"

class AbstractNeuron
{
public:
    AbstractNeuron();

    virtual ~AbstractNeuron();

    virtual number activate(size_t index, std::vector<number> & input) = 0;


    virtual number backpass(size_t index, std::vector<number> & errorsignal,
                                          std::vector<number> &input) = 0;

    virtual number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) = 0;

    virtual number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) = 0;



    virtual number randomWeight(size_t numberOfNeurons = 2);

    virtual number randomWeight(size_t rows, size_t columns);

    virtual bool dropout(number probability);

    virtual void update(size_t epoch);

    virtual number updater(size_t epoch);
};



#endif // ABSTRACTNEURON_H
