#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H

#include "abstractlayer.h"
#include "abstractneuron.h"
#include "generaldata.h"

#include <activationlayerdescription.h>
#include <weightadamoptimization.h>


class ActivationLayer : public AbstractLayer {
public:
    typedef WeightAdamOptimization Updater;
    ActivationLayerDescription desc;
    number outputscaling = 1.0;

    std::vector<AbstractNeuron *> neurons;
    Updater biasUpdater;
    std::vector<Updater::Variable> bias;

    std::vector<GeneralData> data;

public:
    ActivationLayer();

    ActivationLayer(AbstractLayer * prev);

    ActivationLayer(ActivationLayerDescription desc, AbstractLayer * prev);

    ActivationLayer(std::istream & stream, AbstractLayer * prev = nullptr);

    ~ActivationLayer() override;

    void init() override;

    void prepare() override;

    void feedforward() override;

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    std::vector<number> &getInput(size_t index) override{
         return previousLayer->getOutput(index);
    }

    std::vector<size_t> & getActiveInput(size_t index) override {
         return previousLayer->getActiveOutput(index);
    }

    std::vector<number> &getOutput(size_t index) override {
         return this->data[index].output;
    }

    std::vector<size_t> & getActiveOutput(size_t index) override{
        return this->data[index].activeOutput;
    }


    std::vector<number> &getRightErrorSignal(size_t index) override{
        return nextLayer->getLeftErrorSignal(index);
    }

    std::vector<size_t> & getActiveRightErrorSignal(size_t index) override{
        return nextLayer->getActiveLeftErrorSignal(index);
    }

    std::vector<number> &getLeftErrorSignal(size_t index) override {
        return this->data[index].errorSignal;
    }

    std::vector<size_t> & getActiveLeftErrorSignal(size_t index) override {
        return this->data[index].activeErrorSignal;
    }
    };


#endif // ACTIVATIONLAYER_H
