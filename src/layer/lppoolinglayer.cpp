#include "lppoolinglayer.h"

LpPoolingLayer::LpPoolingLayer() : AbstractLayer(){

}

LpPoolingLayer::LpPoolingLayer(AbstractLayer *prev){

}

LpPoolingLayer::LpPoolingLayer(std::istream &stream, AbstractLayer *prev){

}

LpPoolingLayer::~LpPoolingLayer(){

}

void LpPoolingLayer::init()
{

}

void LpPoolingLayer::prepare(){

}

void LpPoolingLayer::feedforward(){

}

void LpPoolingLayer::backprop(){

}

void LpPoolingLayer::update(size_t epoch){

}

void LpPoolingLayer::serialize(std::ostream &out){

}

std::vector<number> &LpPoolingLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}


std::vector<number> &LpPoolingLayer::getOutput(size_t index){
    return this->data[index].output;
}


std::vector<number> &LpPoolingLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}


std::vector<number> &LpPoolingLayer::getLeftErrorSignal(size_t index){
    return this->data[index].errorSignal;
}
