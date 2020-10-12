#include "maxpoolinglayer.h"

MaxPoolingLayer::MaxPoolingLayer() : AbstractLayer(){
    this->type = LayerType::MaxPooling;
    sDebug() << "Neither description nor previous layer provided for MaxPooling Layer";

}

MaxPoolingLayer::MaxPoolingLayer(AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    sDebug() << "No description provided for MaxPooling Layer";
}

MaxPoolingLayer::MaxPoolingLayer(MaxPoolingLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    this->leftActive = previousLayer->rightActive;
    this->rightActive = new SharedActivation();

    // extract description
    this->dimInput = Dimension(desc.dimInput);
    this->dimOutput = Dimension(desc.dimOutput);
    this->patchDimension = Dimension(desc.stride);
    this->channel = desc.channel;


    this->szInput = dimInput.size();
    this->szOutput = this->dimOutput.size();

    this->rightActive->fullsize = this->dimOutput.size()*desc.channel;
    this->rightActive->active.clear();
}

MaxPoolingLayer::MaxPoolingLayer(std::istream &stream,AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::MaxPooling;
    // todo
}

MaxPoolingLayer::~MaxPoolingLayer(){
    // todo
}

void MaxPoolingLayer::init(){
    // todo
}

void MaxPoolingLayer::prepare(){
    this->size = previousLayer->size;
    this->data.resize(this->size);
}

void MaxPoolingLayer::feedforward(){

    for(size_t bz = 0; bz < this->size; bz++){
        auto &input = this->getInput(bz);
        auto &idxInput = this->getActiveInput(bz);

        auto &output = this->getOutput(bz);
        auto &idxOutput = this->getActiveOutput(bz);
        output.resize(this->szOutput*this->channel);
        idxOutput.resize(0);

        auto & trace = this->getActiveLeftErrorSignal(bz); // use error signal sparsity for trace
        trace.resize(this->szOutput*this->channel);

        if(idxInput.size() == 0){
            for (size_t och = 0; och < channel; och++) {
                number *cInput = &input[och * szInput]; // dense input
                number *cOutput = &output[och * szOutput]; // dense output
                size_t *cTrace = &trace[och * szOutput];

                std::vector <size_t> targetcoords;
                size_t patchsize = patchDimension.size();
                std::vector <size_t> coords = dimInput.zeroCoords();
                std::vector <size_t> patchcoords;
                targetcoords = dimInput.zeroCoords();
                for (size_t i = 0; i < szOutput; ++i) {

                    patchcoords = patchDimension.zeroCoords();
                    size_t chindex = dimInput.index(&coords[0]);
                    //sDebug() <<coords <<"\t"<< patchcoords << "\t" << coords << "\t" << chindex;
                    number max = cInput[chindex]; // index = 0;
                    patchDimension.inc(&patchcoords[0]);
                    cTrace[i] = chindex+ och*szInput;
                    if(chindex >= dimInput.size()){
                        sDebug() << patchcoords;
                    }

                    for (size_t p = 1; p < patchsize; ++p) {
                        for (size_t d = 0; d < dimInput.dim; ++d) {
                            targetcoords[d] = patchcoords[d] + coords[d];
                        }
                        chindex = dimInput.index(&targetcoords[0]);
                        //sDebug() <<targetcoords <<"\t"<< patchcoords << "\t" << coords << "\t" << chindex;
                        //if(chindex >= dimInput.size()){
                        //    sDebug() << targetcoords;
                        //}

                        number val = cInput[chindex];
                        if (val > max) {
                            max = val;
                            cTrace[i] = chindex + och*szInput;
                        }
                        patchDimension.inc(&patchcoords[0]);
                    }
                    cOutput[i] = max;
                    dimInput.inc(&coords[0], &patchDimension.gridsize[0]);
                }
            }
        } else  {

    }

    /*
    for(size_t bz = 0; bz < this->size; bz++){
        auto &input = this->getInput(bz);
        auto &idxInput = this->getActiveInput(bz);

        auto &output = this->getOutput(bz);
        auto &idxOutput = this->getActiveOutput(bz);
        output.resize(this->szOutput*this->channel);
        idxOutput.resize(0);

        auto & trace = this->getActiveLeftErrorSignal(bz); // use error signal sparsity for trace
        trace.resize(this->szOutput*this->channel);

        if(idxInput.size() == 0){
            for (size_t och = 0; och < channel; och++) {
                number *cInput = &input[och * szInput]; // dense input
                number *cOutput = &output[och * szOutput]; // dense output
                size_t *cTrace = &trace[och * szOutput];

                std::vector <size_t> targetcoords;
                size_t patchsize = patchDimension.size();
                std::vector <size_t> coords = dimInput.zeroCoords();
                std::vector <size_t> patchcoords;
                targetcoords = dimInput.zeroCoords();
                for (size_t i = 0; i < szOutput; ++i) {

                    patchcoords = patchDimension.zeroCoords();
                    size_t chindex = dimInput.index(&coords[0]);
                    number max = cInput[chindex]; // index = 0;
                    patchDimension.inccoord(&patchcoords[0]);
                    cTrace[i] = chindex+ och*szInput;
                    if(chindex >= dimInput.size()){
                        sDebug() << patchcoords;
                    }

                    for (size_t p = 1; p < patchsize; ++p) {
                        for (size_t d = 0; d < dimInput.dim; ++d) {
                            targetcoords[d] = patchcoords[d] + coords[d];
                        }
                        chindex = dimInput.index(&targetcoords[0]);
                        if(chindex >= dimInput.size()){
                            sDebug() << targetcoords;
                        }

                        number val = cInput[chindex];
                        if (val > max) {
                            max = val;
                            cTrace[i] = chindex;
                        }
                        patchDimension.inccoord(&patchcoords[0]);
                    }
                    cOutput[i] = max;
                    dimInput.inccoordR(&coords[0], &patchDimension.gridsize[0]);
                }
            }
        } else  {
            // make input dense
            std::vector<number> denseInput;
            denseInput.resize(szInput*channel);

            for(size_t i = 0; i < idxInput.size(); i++){
                denseInput[idxInput[i]] = input[i]; // todo real index / channel index
            }

            for (size_t och = 0; och < channel; och++) {
                number *cInput = &denseInput[och * szInput]; // dense input
                number *cOutput = &output[och * szOutput]; // dense output
                size_t *cTrace = &trace[och * szOutput];

                std::vector <size_t> targetcoords;
                size_t patchsize = patchDimension.size();
                std::vector <size_t> coords = dimInput.zeroCoords();
                std::vector <size_t> patchcoords;
                targetcoords = dimInput.zeroCoords();
                for (size_t i = 0; i < szOutput; ++i) {

                    patchcoords = patchDimension.zeroCoords();
                    size_t chindex = dimInput.index(&coords[0]);
                    number max = cInput[chindex];
                    patchDimension.inccoord(&patchcoords[0]);
                    cTrace[i] = chindex + och* szInput;
                    if(chindex >= dimInput.size()){
                        sDebug() << patchcoords;
                    }

                    for (size_t p = 1; p < patchsize; ++p) {
                        for (size_t d = 0; d < dimInput.dim; ++d) {
                            targetcoords[d] = patchcoords[d] + coords[d];
                        }
                        chindex = dimInput.index(&targetcoords[0]);
                        if(chindex >= dimInput.size()){
                            sDebug() << targetcoords;
                        }

                        number val = cInput[chindex];
                        if (val > max) {
                            max = val;
                            cTrace[i] = chindex + och* szInput;
                        }
                        patchDimension.inccoord(&patchcoords[0]);
                    }
                    cOutput[i] = max;
                    dimInput.inccoordR(&coords[0], &patchDimension.gridsize[0]);
                }
            }
        }
    }*/
}
}

void MaxPoolingLayer::backprop(){
    // todo left batch szLeftErrActive

    for(size_t bz = 0; bz < this->size; bz++){
        auto & leftErrorSignal = this->getLeftErrorSignal(bz);
        auto & idxLeftErrorSignal = this->getActiveLeftErrorSignal(bz);
        auto & rightErrorSignal = this->getRightErrorSignal(bz);
        auto & idxRightErrorSignal = this->getActiveRightErrorSignal(bz);

        size_t szRight = idxRightErrorSignal.size();

        leftErrorSignal.resize(idxLeftErrorSignal.size());

        if(szRight == 0){
            for(size_t i = 0; i < rightErrorSignal.size(); i++){
                leftErrorSignal[i] = rightErrorSignal[i];
            }
        } else {
            for(size_t i = 0; i < idxRightErrorSignal.size(); i++){
                idxLeftErrorSignal[i] = idxLeftErrorSignal[idxRightErrorSignal[i]];
                leftErrorSignal[i] = rightErrorSignal[i];
            }
            idxLeftErrorSignal.resize(idxRightErrorSignal.size());
            leftErrorSignal.resize(idxRightErrorSignal.size());
        }


    }
}

void MaxPoolingLayer::update(size_t epoch){
    // nothing to do
}

void MaxPoolingLayer::serialize(std::ostream &out){

}
