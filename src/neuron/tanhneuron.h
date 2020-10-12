#ifndef TANHNEURON_H
#define TANHNEURON_H

#include "abstractneuron.h"



class TanhNeuron : public AbstractNeuron
{
public:
    TanhNeuron();

    ~TanhNeuron() override;
    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;

    number randomWeight(size_t numberOfNeurons = 2) override;
};

#endif // TANHNEURON_H
