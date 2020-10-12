#ifndef SPARSEMAXPOOLING_H
#define SPARSEMAXPOOLING_H

#include "sparseabstractlayer.h"
#include "sparsepoolingdata.h"

#include <maxpoolinglayerdescription.h>
/*
class SparseMaxPooling : SparseAbstractLayer {
    MaxPoolingLayerDescription desc;

    Dimension dimInput;
    Dimension dimOutput;
    Dimension patchDimension;

    size_t channel;

    size_t szInput;
    size_t szOutput;

    std::vector<SparsePoolingData> data;

public:
    SparseMaxPooling();

    SparseMaxPooling(AbstractLayer * prev);

    SparseMaxPooling(MaxPoolingLayerDescription desc, AbstractLayer * prev);

    SparseMaxPooling(std::istream & stream, AbstractLayer * prev);

    ~SparseMaxPooling() override;

    void init() override;

    void prepare() override;

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    std::vector<number> &getInput(size_t index) override;

    number * getInputData(size_t index) override;

    std::vector<number> &getOutput(size_t index) override;

    number * getOutputData(size_t index) override;

    std::vector<number> &getRightErrorSignal(size_t index) override;

    number * getRightErrorSignalData(size_t index) override;

    std::vector<number> &getLeftErrorSignal(size_t index) override;

    number * getLeftErrorSignalData(size_t index) override;



    std::vector<number> &getSparseInput(size_t index);

    std::vector<number> &getSparseOutput(size_t index);

    std::vector<number> &getSparseRightErrorSignal(size_t index);

    std::vector<number> &getSparseLeftErrorSignal(size_t index);

    std::vector<size_t> &getErrorSignalActive(size_t index);

    std::vector<size_t> &getOutputActive(size_t index);

};*/
#endif // SPARSEMAXPOOLING_H
