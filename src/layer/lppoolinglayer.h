#ifndef LPPOOLINGLAYER_H
#define LPPOOLINGLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"
#include "poolingdata.h"



class LpPoolingLayer : public AbstractLayer {
public:
    std::vector<GeneralData> data;
public:
    LpPoolingLayer();

    LpPoolingLayer(AbstractLayer * prev);

    LpPoolingLayer(std::istream & stream, AbstractLayer * prev);

    ~LpPoolingLayer() override;

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

#endif // LPPOOLINGLAYER_H
