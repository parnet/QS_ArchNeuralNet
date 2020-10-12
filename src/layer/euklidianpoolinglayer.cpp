#include "euklidianpoolinglayer.h"

EuklidianPoolingLayer::EuklidianPoolingLayer() : AbstractLayer(){

}

EuklidianPoolingLayer::EuklidianPoolingLayer(AbstractLayer *prev) : AbstractLayer(prev){

}

EuklidianPoolingLayer::EuklidianPoolingLayer(std::istream &stream, AbstractLayer *prev): AbstractLayer(prev){

}

EuklidianPoolingLayer::~EuklidianPoolingLayer(){

}

void EuklidianPoolingLayer::init()
{

}

void EuklidianPoolingLayer::prepare(){

}

void EuklidianPoolingLayer::feedforward(){

}

void EuklidianPoolingLayer::backprop(){

}

void EuklidianPoolingLayer::update(size_t epoch){

}

void EuklidianPoolingLayer::serialize(std::ostream &out){

}

std::vector<number> &EuklidianPoolingLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}

std::vector<number> &EuklidianPoolingLayer::getOutput(size_t index){
    return this->data[index].output;
}

std::vector<number> &EuklidianPoolingLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}


std::vector<number> &EuklidianPoolingLayer::getLeftErrorSignal(size_t index){
    return this->data[index].errorSignal;
}
