#include "stochasticpoolinglayer.h"

StochasticPoolingLayer::StochasticPoolingLayer() : AbstractLayer(){

}

StochasticPoolingLayer::StochasticPoolingLayer(AbstractLayer *prev) : AbstractLayer(prev){

}

StochasticPoolingLayer::StochasticPoolingLayer(std::istream &stream,AbstractLayer *prev):AbstractLayer(prev){

}

StochasticPoolingLayer::~StochasticPoolingLayer(){

}

void StochasticPoolingLayer::init()
{

}

void StochasticPoolingLayer::prepare(){
    // todo
}

void StochasticPoolingLayer::feedforward(){
    // todo
}

void StochasticPoolingLayer::backprop(){
    // todo
}

void StochasticPoolingLayer::update(size_t epoch){
    // nothing to do
}

void StochasticPoolingLayer::serialize(std::ostream &out){
    // todo
}

std::vector<number> &StochasticPoolingLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}


std::vector<number> &StochasticPoolingLayer::getOutput(size_t index){
    return this->data[index].output;
}


std::vector<number> &StochasticPoolingLayer::getRightErrorSignal(size_t index){
    return  nextLayer->getLeftErrorSignal(index);
}
std::vector<number> &StochasticPoolingLayer::getLeftErrorSignal(size_t index){
    return this->data[index].output;
}
