#include "outputlayer.h"

OutputLayer::OutputLayer() : AbstractLayer(){
    this->type = LayerType::Output;
}

OutputLayer::OutputLayer(AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::Output;
    this->leftActive = prev->rightActive;
}

OutputLayer::OutputLayer(OutputLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::Output;
    this->leftActive = prev->rightActive;
}

OutputLayer::OutputLayer(std::istream &stream,AbstractLayer * prev) : AbstractLayer(stream, prev){
    this->type = LayerType::Output;
}

OutputLayer::~OutputLayer(){}

void OutputLayer::init(){
    // todo
}

void OutputLayer::prepare(){
    this->size = previousLayer->size;
    this->data.resize(this->size);
}

void OutputLayer::setTarget(QGPBatch *batch){
   const size_t szBatch = batch->size;
   this->data.resize(szBatch);

   for(size_t i = 0; i < szBatch; ++i){
       this->data[i].setTarget(batch->data[i].target());
   }
}

number OutputLayer::getLoss(size_t index){

    auto & output = this->previousLayer->getOutput(index);
    auto & target = this->data[index].target;

    return -log(1 - fabs(output[1] - target[1])); // todo generalize

}

std::vector<number> OutputLayer::getLoss(){
    std::vector<number> lossvec;
    for(size_t i = 0; i < this->size; i++){
        lossvec[i] = getLoss(i);
    }
    return lossvec;
}

void OutputLayer::feedforward(){
    // nothing to do
}

void OutputLayer::backprop(){
    for(size_t i = 0; i< this->size; i++){
        auto & output = getOutput(i);
        size_t szOutput = output.size();
        auto idxErrorSignal = this->getActiveLeftErrorSignal(i);
        idxErrorSignal.clear();

        this->data[i].leftErrorSignal.resize(szOutput);
        for(size_t sz = 0; sz < szOutput; sz++){
            this->data[i].leftErrorSignal[sz] = output[sz] -  this->data[i].target[sz];
        }
    }
}

void OutputLayer::update(size_t){
    // nothing to do
}

void OutputLayer::serialize(std::ostream &out){

}

void OutputLayer::getTrainingResults(std::vector<std::vector<number> > &output, std::vector<number> &loss){
    output.resize(this->size);
    loss.resize(this->size);
    for(size_t i = 0; i < this->size; i++){
        output[i] = getOutput(i);
        loss[i] = getLoss(i);
    }
}

