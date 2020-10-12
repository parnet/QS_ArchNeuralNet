#ifndef SOFTMAXNEURON_H
#define SOFTMAXNEURON_H

#include "abstractneuron.h"



class SoftmaxNeuron : public AbstractNeuron
{
public:
    SoftmaxNeuron();
    ~SoftmaxNeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;

};

#endif // SOFTMAXNEURON_H
