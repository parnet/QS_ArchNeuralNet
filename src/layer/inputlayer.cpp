#include "inputlayer.h"

InputLayer::InputLayer() : AbstractLayer(){
    this->type = LayerType::Input;
    sDebug() << "Neither description nor previous layer provided for Activation Layer";

}

InputLayer::InputLayer(AbstractLayer *prev) : AbstractLayer(prev) {
    this->type = LayerType::Input;
    sDebug() << "No description provided for Activation Layer";
}

InputLayer::InputLayer(InputLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev){
    this->desc = desc;
    this->type = LayerType::Input;
    this->rightActive = new SharedActivation();
    this->rightActive->fullsize = desc.dimension.size();
}

InputLayer::InputLayer(std::istream &stream, AbstractLayer * prev): AbstractLayer(stream,prev){
    this->type = LayerType::Input;
    this->rightActive = new SharedActivation();
    this->rightActive->fullsize = 224000;
}

InputLayer::~InputLayer(){
    // nothing to do
}

void InputLayer::init(){
    // todo
}

void InputLayer::prepare(){
    // nothing to do
}

void InputLayer::feedforward(){
    // nothing to do
}

void InputLayer::backprop(){
    // nothing to do
}

void InputLayer::update(size_t epoch){
    // nothing to do
}

void InputLayer::updater(size_t epoch){
    // nothing to do
}

void InputLayer::setInput(QGPBatch *batch){

    this->size = batch->size;
    this->data.resize(this->size);

    this->rightActive->active.clear();
    batch->seekActive();
    /* act as a dropout for the batch for the input layer */
    for(size_t i = 0; i < desc.size; i++){
        if(batch->nonzero[i]){
            this->rightActive->active.push_back(i);
        }
    }

    for(size_t bz = 0; bz < this->size ; bz++){
        this->data[bz].setData(batch->data[bz]);
    }
}

void InputLayer::serialize(std::ostream &out){
    // todo
}

