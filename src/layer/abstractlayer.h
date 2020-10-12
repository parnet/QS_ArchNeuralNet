#ifndef ABSTRACTLAYER_H
#define ABSTRACTLAYER_H

#include "environment.h"
#include "layertype.h"
#include "sharedactivation.h"

class AbstractLayer {
public:
    AbstractLayer * previousLayer = nullptr;
    AbstractLayer * nextLayer = nullptr;
    LayerType type = LayerType::Abstract;
    bool training = true;
    bool verbose = false;
    bool calcLeftErrorSignal = true;

    size_t size = 0;
    SharedActivation * rightActive = nullptr;
    SharedActivation * leftActive = nullptr;

public:
    AbstractLayer();

    AbstractLayer(AbstractLayer * prev);

    AbstractLayer(std::istream & stream, AbstractLayer* prev);

    virtual void init() = 0;

    virtual ~AbstractLayer();

    virtual void prepare() = 0;


    virtual void feedforward() = 0;

    virtual void backprop() = 0;

    virtual void update(size_t epoch) = 0;

    virtual void setTraining(bool training);


    virtual void serialize(std::ostream & out) = 0;

    virtual std::vector<number> &getInput(size_t index) = 0;

    virtual std::vector<size_t> &getActiveInput(size_t index) = 0;

    virtual std::vector<number> &getOutput(size_t index) = 0;

    virtual std::vector<size_t> &getActiveOutput(size_t index) = 0;

    virtual std::vector<number> &getRightErrorSignal(size_t index) = 0;

    virtual std::vector<size_t> &getActiveRightErrorSignal(size_t index) = 0;

    virtual std::vector<number> &getLeftErrorSignal(size_t index) = 0;

    virtual std::vector<size_t> &getActiveLeftErrorSignal(size_t index) = 0;


    void displayOutput(size_t index);
    void displayActiveOutput(size_t index);
    void displayInput(size_t index);
    void displayActiveInput(size_t index);
    void displayRightErrorSignal(size_t index);
    void displayActiveRightErrorSignal(size_t index);
    void displayLeftErrorSignal(size_t index);
    void displayActiveLeftErrorSignal(size_t index);


};

#endif // ABSTRACTLAYER_H
