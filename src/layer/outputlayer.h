#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "abstractlayer.h"
#include "outputdata.h"

#include <outputlayerdescription.h>
#include <qgpbatch.h>

class OutputLayer : public AbstractLayer
{
public:
    std::vector<OutputData> data;
public:
    OutputLayer();

    OutputLayer(AbstractLayer * prev);

    OutputLayer(OutputLayerDescription desc, AbstractLayer * prev);

    OutputLayer(std::istream & stream,AbstractLayer * prev);

    ~OutputLayer() override;

    void init() override;

    void prepare() override;

    void setTarget(QGPBatch * batch);

    number getLoss(size_t index);

    std::vector<number> getLoss();

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    void getTrainingResults(std::vector<std::vector<number>> & output, std::vector<number> &loss );

    std::vector<number> &getInput(size_t index) override{
        return previousLayer->getOutput(index);
    }

    std::vector<size_t> & getActiveInput(size_t index) override{
        return previousLayer->getActiveOutput(index);
    }

    std::vector<number> &getOutput(size_t index) override {
        return  previousLayer->getOutput(index);
    }

    std::vector<size_t> & getActiveOutput(size_t index) override{
        return previousLayer->getActiveOutput(index);
    }

    std::vector<number> &getRightErrorSignal(size_t index) override {
        return this->data[index].leftErrorSignal;
    }

    std::vector<size_t> & getActiveRightErrorSignal(size_t index) override{
        return this->data[index].activeErrorSignal;
    }

    std::vector<number> &getLeftErrorSignal(size_t index) override {
        return this->data[index].leftErrorSignal;
    }

    std::vector<size_t> & getActiveLeftErrorSignal(size_t index) override{
        return this->data[index].activeErrorSignal;
    }

};

#endif // OUTPUTLAYER_H
