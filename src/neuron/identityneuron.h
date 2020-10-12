#ifndef IDENTITYNEURON_H
#define IDENTITYNEURON_H

#include "abstractneuron.h"



class IdentityNeuron : public AbstractNeuron
{
public:
    IdentityNeuron();

    ~IdentityNeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal,
                                  std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                    size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                    size_t inputindex, std::vector<number> & input,
                    size_t outputindex, std::vector<number> & output) override;
};

#endif // IDENTITYNEURON_H
