#ifndef STOCHASTICPOOLINGLAYER_H
#define STOCHASTICPOOLINGLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"
#include "poolingdata.h"



class StochasticPoolingLayer : public AbstractLayer{
public:
        std::vector<GeneralData> data;
public:
    StochasticPoolingLayer();

    StochasticPoolingLayer(AbstractLayer * prev);

    StochasticPoolingLayer( std::istream & stream, AbstractLayer * prev);

    ~StochasticPoolingLayer() override;

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

#endif // STOCHASTICPOOLINGLAYER_H
