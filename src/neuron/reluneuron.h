#ifndef RELUNEURON_H
#define RELUNEURON_H

#include "abstractneuron.h"



class ReLUNeuron : public AbstractNeuron
{
public:
    ReLUNeuron();

    ~ReLUNeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;
};

#endif // RELUNEURON_H
