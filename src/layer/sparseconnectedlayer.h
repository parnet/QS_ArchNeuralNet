#ifndef SPARSECONNECTEDLAYER_H
#define SPARSECONNECTEDLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"


class SparseConnectedLayer : public AbstractLayer
{
public:
    std::vector<GeneralData> data;
public:
    SparseConnectedLayer();

    SparseConnectedLayer(AbstractLayer * prev);

    SparseConnectedLayer(std::istream & stream,AbstractLayer * prev);

    ~SparseConnectedLayer() override;

    void init() override;

    void prepare() override;

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    std::vector<number> &getInput(size_t index) override;

    std::vector<number> &getOutput(size_t index) override;

    std::vector<number> &getRightErrorSignal(size_t index) override;

    std::vector<number> &getLeftErrorSignal(size_t index) override;


};

#endif // SPARSECONNECTEDLAYER_H
