#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"

#include <fullyconnecteddescription.h>
#include <weightadamoptimization.h>


class FullyConnectedLayer : public AbstractLayer{
public:
    FullyConnectedDescription desc;
    typedef WeightAdamOptimization Updater;

    std::vector<GeneralData> data;

    Updater updater;


    std::vector<std::vector<Updater::Variable>> connections;
public:
    FullyConnectedLayer();

    FullyConnectedLayer(AbstractLayer * prev);

    FullyConnectedLayer(FullyConnectedDescription desc, AbstractLayer *prev);

    FullyConnectedLayer(std::istream & stream,AbstractLayer * prev );

    ~FullyConnectedLayer() override;

    void init() override;

    void prepare() override;

    void feedforward() override;

    void calcInputChanges();

    void calcWeightChanges();

    void backprop() override;

    void update(size_t epoch) override;

    void serialize(std::ostream & out) override;

    void toMatlab(std::string path);

    std::vector<number> &getInput(size_t index) override {
        return previousLayer->getOutput(index);
    }

    std::vector<size_t> & getActiveInput(size_t index) override{
        return previousLayer->getActiveOutput(index);
    }

    std::vector<number> &getOutput(size_t index)override {
        return this->data[index].output;
    }

    std::vector<size_t> & getActiveOutput(size_t index) override {
        return this->data[index].activeOutput;
    }


    std::vector<number> &getRightErrorSignal(size_t index) override {
        return nextLayer->getLeftErrorSignal(index);
    }

    std::vector<size_t> & getActiveRightErrorSignal(size_t index) override{
        return nextLayer->getActiveLeftErrorSignal(index);
    }


    std::vector<number> &getLeftErrorSignal(size_t index) override {
        return this->data[index].errorSignal;
    }

    std::vector<size_t> & getActiveLeftErrorSignal(size_t index) override{
        return this->data[index].activeErrorSignal;
    }

    void displayOutput();


    void displayWeights();

    void displayWeightChanges();

};


#endif // FULLYCONNECTEDLAYER_H
