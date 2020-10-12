#include "averagepoolinglayer.h"

AveragePoolingLayer::AveragePoolingLayer() : AbstractLayer(){
    // todo
}

AveragePoolingLayer::AveragePoolingLayer(AbstractLayer *prev) : AbstractLayer(prev){
    // todo
}

AveragePoolingLayer::AveragePoolingLayer(std::istream &stream, AbstractLayer *prev) : AbstractLayer(prev){
    // todo
}

AveragePoolingLayer::~AveragePoolingLayer(){
    // todo
}

void AveragePoolingLayer::init(){
    // todo
}

void AveragePoolingLayer::prepare(){
    // todo
}

void AveragePoolingLayer::feedforward(){
    // todo
    /*for (size_t och = 0; och < this->driver.outChannel; och++) {

        number *cPoolingActivated = &input[och * this->driver->szPooling];
        number *cOutput = &output[och * this->driver->szOutput];
        size_t *cTrace = &trace[och * this->driver->szOutput];


        std::vector <size_t> targetcoords;
        size_t patchsize = this->driver->patchDimension.size();
        std::vector <size_t> coords = this->driver->dimPooling.zeroCoords();
        std::vector <size_t> patchcoords;
        targetcoords = this->driver->dimPooling.zeroCoords();
        for (size_t i = 0; i < this->driver->szOutput; ++i) {

            patchcoords = this->driver->dimPooling.zeroCoords();
            size_t index = this->driver->dimPooling.index(&coords[0]);
            number max = cPoolingActivated[index]; // index = 0;
            this->driver->patchDimension.inccoord(&patchcoords[0]);
            cTrace[i] = index;
            if(index >= this->driver->dimPooling.size()){
                sDebug() << patchcoords;
            }

            for (size_t p = 1; p < patchsize; ++p) {
                for (size_t d = 0; d < this->driver->dimInput.dim; ++d) {
                    targetcoords[d] = patchcoords[d] + coords[d];
                }
                index = this->driver->dimPooling.index(&targetcoords[0]);
                if(index >= this->driver->dimPooling.size()){
                    sDebug() << targetcoords;
                }

                number val = cPoolingActivated[index];
                max += val;
                if (val > max) {
                    cTrace[i] = index;
                }
                this->driver->patchDimension.inccoord(&patchcoords[0]);
            }
            cOutput[i] = max;
            this->driver->dimPooling.inccoordR(&coords[0], &this->driver->patchDimension.gridsize[0]);
            // sDebug() << coords;
        }
    }*/

}

void AveragePoolingLayer::backprop(){
    // todo
}

void AveragePoolingLayer::update(size_t epoch){
    // todo
}

void AveragePoolingLayer::serialize(std::ostream &out){
    // todo
}

std::vector<number> &AveragePoolingLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}

std::vector<number> &AveragePoolingLayer::getOutput(size_t index){
    return data[index].output;
}

std::vector<number> &AveragePoolingLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}


std::vector<number> &AveragePoolingLayer::getLeftErrorSignal(size_t index){
    return this->data[index].errorSignal;
}

