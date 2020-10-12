#include "sparsemaxpooling.h"
/*
SparseMaxPooling::SparseMaxPooling() : SparseAbstractLayer(){
    this->type = LayerType::MaxPooling;
    sDebug() << "Neither description nor previous layer provided for MaxPooling Layer";
}

SparseMaxPooling::SparseMaxPooling(AbstractLayer *prev) : SparseAbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    sDebug() << "No description provided for MaxPooling Layer";
}

SparseMaxPooling::SparseMaxPooling(MaxPoolingLayerDescription desc, AbstractLayer *prev) : SparseAbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    this->leftActive = previousLayer->rightActive;
    this->rightActive = new SharedActivation();

    this->dimInput = Dimension(desc.dimInput);
    this->dimOutput = Dimension(desc.dimOutput);
    this->patchDimension = Dimension(desc.stride);

    this->szInput = this->dimInput.size();
    this->szOutput = this->dimOutput.size();

    this->channel = desc.channel;
    this->rightActive->fullsize = this->dimOutput.size()*desc.channel;
    this->rightActive->active.resize(this->rightActive->fullsize);

    for(size_t i = 0; i < this->rightActive->fullsize; i++){
        this->rightActive->active[i] = i;
    }
}

SparseMaxPooling::SparseMaxPooling(std::istream &stream, AbstractLayer *prev) : SparseAbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    //todo
}

SparseMaxPooling::~SparseMaxPooling(){
    // todo
}

void SparseMaxPooling::init(){
    // todo
}

void SparseMaxPooling::prepare(){
    this->size = previousLayer->size;
    this->data.resize(this->size);
}

void SparseMaxPooling::feedforward(){
    for(size_t bz = 0; bz < this->size; bz++){
        auto & input = this->getInput(bz);
        auto & output = this->data[bz].output;
        auto & trace = this->data[bz].activeErrorSignal;

        trace.resize(this->szOutput*this->channel);
        output.resize(this->szOutput*this->channel);

        for (size_t och = 0; och < channel; och++) {

            number *cInput = &input[och * szInput];
            number *cOutput = &output[och * szOutput];
            size_t *cTrace = &trace[och * szOutput];

            std::vector <size_t> targetcoords;
            size_t patchsize = patchDimension.size();
            std::vector <size_t> coords = dimInput.zeroCoords();
            std::vector <size_t> patchcoords;
            targetcoords = dimInput.zeroCoords();
            for (size_t i = 0; i < szOutput; ++i) {

                patchcoords = dimInput.zeroCoords();
                size_t index = dimInput.index(&coords[0]);
                number max = cInput[index]; // index = 0;
                patchDimension.inccoord(&patchcoords[0]);
                cTrace[i] = index;
                if(index >= dimInput.size()){
                    sDebug() << patchcoords;
                }

                for (size_t p = 1; p < patchsize; ++p) {
                    for (size_t d = 0; d < dimInput.dim; ++d) {
                        targetcoords[d] = patchcoords[d] + coords[d];
                    }
                    index = dimInput.index(&targetcoords[0]);
                    if(index >= dimInput.size()){
                        sDebug() << targetcoords;
                    }

                    number val = cInput[index];
                    if (val > max) {
                        max = val; // index = p
                        cTrace[i] = index;
                    }
                    patchDimension.inccoord(&patchcoords[0]);
                }
                cOutput[i] = max;
                dimInput.inccoordR(&coords[0], &patchDimension.gridsize[0]);
                // sDebug() << coords;
            }
        }
    }
}

void SparseMaxPooling::backprop(){
    for(size_t bz = 0; bz < this->size; bz++){
        data[bz].leftErrorSignal = this->getRightErrorSignal(bz);
    }
}

void SparseMaxPooling::update(size_t epoch){
    // nothing to do
}

void SparseMaxPooling::serialize(std::ostream &out){
    // todo
}

std::vector<number> &SparseMaxPooling::getInput(size_t index){
    return previousLayer->getOutput(index);
}

number *SparseMaxPooling::getInputData(size_t index){
    return previousLayer->getOutputData(index);
}

std::vector<number> &SparseMaxPooling::getOutput(size_t index){
    return this->data[index].output;
}

number *SparseMaxPooling::getOutputData(size_t index){

}

std::vector<number> &SparseMaxPooling::getRightErrorSignal(size_t index){

}

number *SparseMaxPooling::getRightErrorSignalData(size_t index){

}

std::vector<number> &SparseMaxPooling::getLeftErrorSignal(size_t index){

}

number *SparseMaxPooling::getLeftErrorSignalData(size_t index){

}

std::vector<number> &SparseMaxPooling::getSparseInput(size_t index){

}

std::vector<number> &SparseMaxPooling::getSparseOutput(size_t index){
    return this->data[index].output;
}

std::vector<number> &SparseMaxPooling::getSparseRightErrorSignal(size_t index){

}

std::vector<number> &SparseMaxPooling::getSparseLeftErrorSignal(size_t index){
    return this->data[index].leftErrorSignal;
}

std::vector<size_t> &SparseMaxPooling::getErrorSignalActive(size_t index){
    return this->data[index].activeErrorSignal;
}

std::vector<size_t> &SparseMaxPooling::getOutputActive(size_t index){
    return this->data[index].activeOutput;
}
*/
