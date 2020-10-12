#ifndef TESTLAYER_H
#define TESTLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"



class TestLayer : public AbstractLayer{
public:
    std::vector<GeneralData> data;
    std::vector<GeneralData> compare;
public:
    TestLayer();

    TestLayer(AbstractLayer * layer);

    ~TestLayer() override;

    void init() override;

    void prepare() override;

    void setSize(size_t size);

    void setRightVectorSize(size_t size);

    void setLeftVectorSize(size_t size);

    void setRightVectorSize(size_t size, size_t );

    void setLeftActive(std::vector<size_t>  active);

    void setRightActive(std::vector<size_t> active);

    void setTraining(bool training) override;

    void update(size_t epoch) override;

    void feedforward() override;

    void backprop() override;

    void serialize(std::ostream &out) override;


    bool controlInput();

    bool controlErrorSignal();


    std::vector<number> createUniformVector(size_t size, number p_from, number p_to);

    void setOutput(size_t index, std::vector<number> value, std::vector<size_t> active = {});

    void setExpectedInput(size_t index, std::vector<number> value, std::vector<size_t> active = {});

    void setErrorSignal(size_t index, std::vector<number> value, std::vector<size_t> active = {});

    void setExpectedErrorSignal(size_t index, std::vector<number> value, std::vector<size_t> active = {});

    std::vector<number> & getInput(size_t index) override{
        return previousLayer->getOutput(index);
    }

    std::vector<size_t> & getActiveInput(size_t index) override {
        return previousLayer->getActiveOutput(index);
    }

    std::vector<number> & getOutput(size_t index) override{
        return this->data[index].output;
    }

    std::vector<size_t> & getActiveOutput(size_t index) override {
        return this->data[index].activeOutput;
    }

    std::vector<number> & getRightErrorSignal(size_t index) override {
        return nextLayer->getLeftErrorSignal(index);
    }

    std::vector<size_t> & getActiveRightErrorSignal(size_t index) override {
        return nextLayer->getActiveLeftErrorSignal(index);
    }

    std::vector<number> & getLeftErrorSignal(size_t index) override {
        return this->data[index].errorSignal;
    }

    std::vector<size_t> & getActiveLeftErrorSignal(size_t index) override{
        return this->data[index].activeErrorSignal;
    }
};

#endif // TESTLAYER_H
