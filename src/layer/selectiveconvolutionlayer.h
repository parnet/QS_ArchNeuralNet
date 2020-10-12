#ifndef SELECTIVECONVOLUTIONLAYER_H
#define SELECTIVECONVOLUTIONLAYER_H

#include "abstractlayer.h"
#include "generaldata.h"



class SelectiveConvolutionLayer : public AbstractLayer{
public:
    std::vector<GeneralData> data;
public:
    SelectiveConvolutionLayer();

    SelectiveConvolutionLayer(AbstractLayer * prev);

    SelectiveConvolutionLayer(std::istream & stream, AbstractLayer * prev);

    ~SelectiveConvolutionLayer() override;

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

#endif // SELECTIVECONVOLUTIONLAYER_H
