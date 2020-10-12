#ifndef NORMALIZATIONLAYER_H
#define NORMALIZATIONLAYER_H

#include "abstractlayer.h"
#include "normalizationdata.h"
#include "normalizationdriver.h"




class NormalizationLayer : public AbstractLayer{
public:
    NormalizationDriver driver;
    std::vector<NormalizationData> data;

    std::vector<number> aMean; // dimension: size of active neurons
    std::vector<number> aVariance; // dimension: size of active neurons


public:
    NormalizationLayer();

    NormalizationLayer(AbstractLayer * prev);

    NormalizationLayer(std::istream & stream, AbstractLayer * prev);

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

#endif // NORMALIZATIONLAYER_H
