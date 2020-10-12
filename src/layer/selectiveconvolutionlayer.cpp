#include "selectiveconvolutionlayer.h"

SelectiveConvolutionLayer::SelectiveConvolutionLayer() : AbstractLayer(){

}

SelectiveConvolutionLayer::SelectiveConvolutionLayer(AbstractLayer *prev){

}

SelectiveConvolutionLayer::SelectiveConvolutionLayer(std::istream &stream,AbstractLayer *prev):AbstractLayer(stream,prev){

}

SelectiveConvolutionLayer::~SelectiveConvolutionLayer(){

}

void SelectiveConvolutionLayer::init(){

}

void SelectiveConvolutionLayer::prepare(){

}

void SelectiveConvolutionLayer::feedforward(){

}

void SelectiveConvolutionLayer::backprop(){

}

void SelectiveConvolutionLayer::update(size_t epoch){

}

void SelectiveConvolutionLayer::serialize(std::ostream &out){

}

std::vector<number> &SelectiveConvolutionLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}


std::vector<number> &SelectiveConvolutionLayer::getOutput(size_t index){
    return this->data[index].output;
}

std::vector<number> &SelectiveConvolutionLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}

std::vector<number> &SelectiveConvolutionLayer::getLeftErrorSignal(size_t index){
    return this->data[index].errorSignal;
}
