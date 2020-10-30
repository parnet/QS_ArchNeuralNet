#ifndef SCONVOLUTIONALLAYER_H
#define SCONVOLUTIONALLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"

#include <convolutionlayerdescription.h>
#include <weightadamoptimization.h>
#include <weightgradientdescent.h>

class SFilter {
public:
    typedef WeightAdamOptimization KernelUpdater;
    typedef WeightAdamOptimization BiasUpdater;
public:
    std::vector<KernelUpdater::Variable> kernel;
    BiasUpdater::Variable bias;
    KernelUpdater kernelUpdater;
    BiasUpdater biasUpdater;
    number learningrateKernel = 0.001; // todo move to Updater // todo fix value
    number learningRateBias = 0.001; // todo move to Updater
};



class SConvolutionalLayer : public AbstractLayer {
public:
    std::vector<GeneralData> data;

    ConvolutionLayerDescription desc;


    size_t inChannel;
    size_t outChannel;


    Dimension dimInput;
    Dimension dimOutput;
    Dimension dimKernel;


    size_t szInput;
    size_t szOutput;
    size_t szKernel;


    std::vector<SFilter> filter;
    std::vector<std::vector<size_t>> indexmap;
    std::vector<std::vector<size_t>> dualmap;

public:
    SConvolutionalLayer();

    SConvolutionalLayer(AbstractLayer * prev);

    SConvolutionalLayer(ConvolutionLayerDescription desc, AbstractLayer * prev);

    SConvolutionalLayer(std::istream & stream,AbstractLayer * prev);

    ~SConvolutionalLayer() override;

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

    std::vector<size_t> &getActiveOutput(size_t index) override
    {
        return this->data[index].activeOutput;
    }


    std::vector<number> &getRightErrorSignal(size_t index) override {
        return nextLayer->getLeftErrorSignal(index);
    }

    std::vector<size_t> &getActiveRightErrorSignal(size_t index) override
    {
        return nextLayer->getActiveLeftErrorSignal(index);
    }

    std::vector<number> &getLeftErrorSignal(size_t index) override {
        return this->data[index].errorSignal;
    }

    virtual std::vector<size_t> &getActiveLeftErrorSignal(size_t index) override
    {
        return this->data[index].activeErrorSignal;
    }


public:
    void calcKernelChanges(size_t bz);

    void calcBiasChanges(size_t bz);

    void calcInputChanges(size_t bz);

    void setRandom();

    void createIndexmap(ConvolutionLayerDescription desc);

    void createDualmap();


    void displayIndexmap();

    void displayDualmap();

    void displayKernel();

    void displayKernelChanges();

    void displayBias();

    void displayBiasChanges();

};
#endif // SCONVOLUTIONALLAYER_H
