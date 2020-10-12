#ifndef CONVOLUTIONALDRIVER_H
#define CONVOLUTIONALDRIVER_H

#include "environment.h"
#include "convolutionlayerdescription.h"
#include <dimension.h>
#include <weightadamoptimization.h>
#include <weightgradientdescent.h>

class Filter {
public:
    typedef WeightAdamOptimization KernelUpdater;
    typedef WeightGradientDescent BiasUpdater;
public:
    std::vector<KernelUpdater::Variable> kernel;
    BiasUpdater::Variable bias;
    KernelUpdater kernelUpdater;
    BiasUpdater biasUpdater;
    number learningrateKernel = 0.001; // todo move to Updater
    number learningRateBias = 0.0001; // todo move to Updater
};

class ConvolutionalDriver {
public:

public:
    bool calcLeftErrorSignal = true;

    size_t inChannel;
    size_t outChannel;

    Dimension dimInput;
    Dimension dimOutput;
    Dimension dimKernel;

    size_t szInput;
    size_t szOutput;
    size_t szKernel;

    std::vector<Filter> filter;
    std::vector<std::vector<size_t>> indexmap;

    ConvolutionLayerDescription desc;
public:
    ConvolutionalDriver();
    ConvolutionalDriver(ConvolutionLayerDescription desc);
    ConvolutionalDriver(std::istream &file);


    void setRandom();

    void createIndexmap(ConvolutionLayerDescription desc);

    void update(size_t epoch);

    void displayIndexmap();

    void displayKernel();

    void displayKernelChanges();

    void displayBias();

    void displayBiasChanges();

    void setPredefinedKernel();

    void serialize(std::ostream & out);



};

#endif // CONVOLUTIONALDRIVER_H
