#ifndef SUPERVISOR_H
#define SUPERVISOR_H

#include "neuralnet.h"



class Supervisor{
public:
        NeuralNet * neuralNet = nullptr;
        bool validateonly = false;
        Module * module = nullptr;

        //std::string pathDataset;

        size_t szValidation;
        size_t szEpoch;
        size_t szTraining;
        size_t szBatch;

        bool running = false;
        size_t curBatch = 0;
        size_t curEpoch = 0;
        size_t curTraining = 0;
        size_t curValidation = 0;


        std::vector<size_t> statTotalTraining;
        std::vector<size_t> statCorrectTraining;

        std::vector<size_t> statTotalValidation;
        std::vector<size_t> statCorrectValidation;

        std::vector<number> statLossTraining;
        std::vector<number> statLossValidation;

public:
    Supervisor();

    Supervisor(Module * module);

    ~Supervisor();

    void trainingSingle();

    void validateSingle();

    void training();

    void validate();

    void runTraining();

    void runValidation();

    void setBatchSize(size_t number);

    void setEpoch(size_t szEpoch);

    void setTopology(Topology topology);

    void toFile(const std::string & filepath);

private:
    void clearStats();

};

#endif // SUPERVISOR_H
