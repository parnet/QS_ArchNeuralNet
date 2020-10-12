#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "abstractlayer.h"
#include "inputdata.h"

#include <inputlayerdescription.h>
#include <qgpbatch.h>


class InputLayer : public AbstractLayer {
public:
    InputLayerDescription desc;
    std::vector<InputData> data; // todo replace by Batch?

public:
    InputLayer();

    InputLayer(AbstractLayer * prev);

    InputLayer(InputLayerDescription desc, AbstractLayer * prev);

    InputLayer(std::istream & stream, AbstractLayer * prev = nullptr);

    ~InputLayer() override;

    void init() override;

    void setInput(QGPBatch * batch);

    void prepare() override;

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void updater(size_t epoch);

    void serialize(std::ostream & out) override;

    std::vector<number> &getInput(size_t index) override {
        return this->data[index].output;
    }

    std::vector<size_t> &getActiveInput(size_t index) override {
        return this->data[index].active;
    }

    std::vector<number> &getOutput(size_t index) override {
        return this->data[index].output;
    }

    std::vector<size_t> & getActiveOutput(size_t index) override {
        return this->data[index].active;
    }

    std::vector<number> &getRightErrorSignal(size_t index) override {
        // nothing to do
    }

    std::vector<size_t> & getActiveRightErrorSignal(size_t index) override{

    }

    std::vector<number> &getLeftErrorSignal(size_t index) override {
        // nothing to do
    }

    std::vector<size_t> & getActiveLeftErrorSignal(size_t index) override {

    }

};

#endif // INPUTLAYER_H
