#include "combinedlayer.h"

CombinedLayer::CombinedLayer() : AbstractLayer(){
    // todo
}

CombinedLayer::CombinedLayer(AbstractLayer *prev):AbstractLayer(prev){
    // todo
}

CombinedLayer::CombinedLayer(std::istream &stream, AbstractLayer *prev):AbstractLayer(stream,prev){
    // todo
}

CombinedLayer::~CombinedLayer(){
    // todo
}

void CombinedLayer::init(){
    // todo
}

void CombinedLayer::prepare(){
    // todo
}

void CombinedLayer::feedforward(){
    // todo
}

void CombinedLayer::backprop(){
    // todo
}

void CombinedLayer::update(size_t epoch){
    // todo
}

void CombinedLayer::serialize(std::ostream &out){
    // todo
}

std::vector<number> &CombinedLayer::getInput(size_t index){

}


std::vector<number> &CombinedLayer::getOutput(size_t index){

}


std::vector<number> &CombinedLayer::getRightErrorSignal(size_t index){

}


std::vector<number> &CombinedLayer::getLeftErrorSignal(size_t index){
}
