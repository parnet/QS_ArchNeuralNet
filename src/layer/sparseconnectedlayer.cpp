#include "sparseconnectedlayer.h"

SparseConnectedLayer::SparseConnectedLayer(){

}

SparseConnectedLayer::SparseConnectedLayer(AbstractLayer *prev):AbstractLayer(prev){

}

SparseConnectedLayer::SparseConnectedLayer(std::istream &stream, AbstractLayer *prev) : AbstractLayer(stream, prev){

}

SparseConnectedLayer::~SparseConnectedLayer(){

}

void SparseConnectedLayer::init()
{

}

void SparseConnectedLayer::prepare(){
    // todo
}

void SparseConnectedLayer::feedforward(){
    // todo
}

void SparseConnectedLayer::backprop(){
    // todo
}

void SparseConnectedLayer::update(size_t epoch){
    // todo
}

void SparseConnectedLayer::serialize(std::ostream &out){
    // todo
}

std::vector<number> &SparseConnectedLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}

std::vector<number> &SparseConnectedLayer::getOutput(size_t index){
    return this->data[index].output;
}

std::vector<number> &SparseConnectedLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}


std::vector<number> &SparseConnectedLayer::getLeftErrorSignal(size_t index){
    return this->data[index].errorSignal;
}
