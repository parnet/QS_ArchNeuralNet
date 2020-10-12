#ifndef BIASNEURON_H
#define BIASNEURON_H

#include "abstractneuron.h"



class BiasNeuron : public AbstractNeuron
{
public:
    BiasNeuron();

    ~BiasNeuron() override;

    number activate(size_t index, std::vector<number> & input) override;

    number backpass(size_t index, std::vector<number> & errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output) override;



    bool dropout(number probability) override;


};

#endif // BIASNEURON_H
