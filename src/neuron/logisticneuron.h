#ifndef LOGISTICNEURON_H
#define LOGISTICNEURON_H

#include "abstractneuron.h"



class LogisticNeuron : public AbstractNeuron
{
public:
    LogisticNeuron();

    ~LogisticNeuron() override;

    number activate(size_t index, std::vector<number> &input) override;

    number backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input) override;

    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input);


    number backpass(size_t errorindex, std::vector<number> & errorsignal,
                            size_t inputindex, std::vector<number> & input,
                            size_t outputindex, std::vector<number> & output);

};

#endif // LOGISTICNEURON_H
