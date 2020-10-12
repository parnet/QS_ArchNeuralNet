#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H


#include "abstractlayer.h"
#include "generaldata.h"
#include "poolingdata.h"

#include <dimension.h>
#include <maxpoolinglayerdescription.h>



class MaxPoolingLayer : public AbstractLayer {
    MaxPoolingLayerDescription desc;

    Dimension dimInput;
    Dimension dimOutput;
    Dimension patchDimension;

    size_t channel;

    size_t szInput;
    size_t szOutput;

    std::vector<GeneralData> data;

public:
    MaxPoolingLayer();

    MaxPoolingLayer(AbstractLayer * prev);

    MaxPoolingLayer(MaxPoolingLayerDescription desc, AbstractLayer * prev);

    MaxPoolingLayer(std::istream & stream, AbstractLayer * prev);

    ~MaxPoolingLayer() override;

    void init() override;

    void prepare() override;

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    std::vector<number> &getInput(size_t index) override {
        return previousLayer->getOutput(index);
    }


    std::vector<size_t> &getActiveInput(size_t index) override {
        return previousLayer->getActiveOutput(index);
    }


    std::vector<number> &getOutput(size_t index) override {
        return this->data[index].output;
    }


    std::vector<size_t> &getActiveOutput(size_t index) override{
        return this->data[index].activeOutput;
    }


    std::vector<number> &getRightErrorSignal(size_t index) override {
         return nextLayer->getLeftErrorSignal(index);
    }


    std::vector<size_t> &getActiveRightErrorSignal(size_t index) override {
        return nextLayer->getActiveLeftErrorSignal(index);
    }

    std::vector<number> &getLeftErrorSignal(size_t index) override {
        return this->data[index].errorSignal;
    }


    std::vector<size_t> &getActiveLeftErrorSignal(size_t index) override
    {
        return this->data[index].activeErrorSignal;
    }


};

#endif // POOLINGLAYER_H
