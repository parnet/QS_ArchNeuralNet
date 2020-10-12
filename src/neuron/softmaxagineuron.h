#ifndef SOFTMAXAGINEURON_H
#define SOFTMAXAGINEURON_H

#include "abstractneuron.h"



class SoftmaxAGINeuron : public AbstractNeuron
{
public:
    SoftmaxAGINeuron();
    ~SoftmaxAGINeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;

};

#endif // SOFTMAXAGINEURON_H
