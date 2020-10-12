#ifndef LEAKYRELUNEURON_H
#define LEAKYRELUNEURON_H

#include "abstractneuron.h"



class LeakyReLUNeuron : public AbstractNeuron
{
public:
    number slope = 1e-2;
public:
    LeakyReLUNeuron();

    ~LeakyReLUNeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;

    number randomWeight(size_t numberOfNeurons = 2) override;
};

#endif // LEAKYRELUNEURON_H
