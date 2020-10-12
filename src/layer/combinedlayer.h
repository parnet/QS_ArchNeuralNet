#ifndef COMBINEDLAYER_H
#define COMBINEDLAYER_H

#include "abstractlayer.h"



class CombinedLayer : public AbstractLayer
{
public:
    std::vector<AbstractLayer*> layer;
public:
    CombinedLayer();

    CombinedLayer(AbstractLayer * prev);

    CombinedLayer( std::istream & stream,AbstractLayer * prev);

    ~CombinedLayer() override;

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

#endif // COMBINEDLAYER_H
