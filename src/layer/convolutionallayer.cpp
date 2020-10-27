#include "convolutionallayer.h"
#include "randomdevice.h"

ConvolutionalLayer::ConvolutionalLayer() : AbstractLayer(){
    this->type = LayerType::Convolution;
    sDebug() << "Neither description nor previous layer provided for Convolutional Layer";
}


ConvolutionalLayer::ConvolutionalLayer(AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::Convolution;
    sDebug() << "No description provided for Convolutional Layer";
}

ConvolutionalLayer::ConvolutionalLayer(ConvolutionLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev)
  , dimInput(desc.dimInput),
      dimOutput(desc.dimOutput), dimKernel(desc.dimKernel){
    this->type = LayerType::Convolution;

    this->desc = desc;

    this->leftActive = prev->rightActive;
    this->rightActive = new SharedActivation();

    this->rightActive->fullsize = dimOutput.size()*desc.outChannel;


    this->inChannel = desc.inChannel;
    this->outChannel = desc.outChannel;


    szInput = dimInput.size();
    szOutput = dimOutput.size();
    szKernel = dimKernel.size();

    filter.resize(outChannel);
    for (size_t och = 0; och < outChannel; och++) {
        filter[och].kernel.resize(szKernel * inChannel);
        filter[och].bias.weight = 0;
        filter[och].bias.gradient = 0;
    }

    this->setRandom();
    this->createIndexmap(desc);

    // todo learningrate kernel
    // todo learningrate bias
    // todo inputlayer calc
}

ConvolutionalLayer::ConvolutionalLayer( std::istream &stream, AbstractLayer *prev):AbstractLayer(stream, prev){
    // todo
}

ConvolutionalLayer::~ConvolutionalLayer(){
    // todo
}

void ConvolutionalLayer::init(){
    // todo
}

void ConvolutionalLayer::prepare(){
    this->size = previousLayer->size;
    this->data.clear();
    this->data.resize(this->size, GeneralData());
}

void ConvolutionalLayer::feedforward(){
    for(size_t bz = 0; bz < this->size; bz++){
        auto input = previousLayer->getOutput(bz);
        auto idxInput = previousLayer->getActiveOutput(bz);
        data[bz].output.resize(szOutput*outChannel);
        std::fill(&data[bz].output[0], &data[bz].output[szOutput*outChannel], 0.0 );

        if(idxInput.size() != 0){
            // make it dense
            std::vector<number> denseInput;
            denseInput.resize(this->leftActive->fullsize);
            for(size_t i = 0; i < idxInput.size(); i++){
                denseInput[idxInput[i]] = input[i];
            }

            for (size_t och = 0; och < outChannel; och++) {
                auto &kernel = filter[och].kernel;
                auto &bias = filter[och].bias;
                number *cPooling = &data[bz].output[och * szOutput];
                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &denseInput[ch * szInput];
                    auto *cKernel = &kernel[ch * szKernel];
                    for (size_t i = 0; i < szOutput; ++i) {
                        number sum = 0.0;
                        std::vector <size_t> &row = indexmap[i];
                        for (size_t k = 0; k < szKernel; ++k) {
                            if (row[k] == szInput) {
                                continue;
                            }
                            sum += cKernel[k].weight * cInput[row[k]];
                        }
                        cPooling[i] += sum; // todo check close 0 to enforce sparsity ?
                    }
                }

                for (size_t i = 0; i < szOutput; ++i) {
                    cPooling[i] += bias.weight; // apply bias
                }
            }

        } else { // input is already dense
            for (size_t och = 0; och < outChannel; och++) {
                auto &kernel = filter[och].kernel;
                auto &bias = filter[och].bias;
                number *cPooling = &data[bz].output[och * szOutput];
                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &input[ch * szInput];
                    auto *cKernel = &kernel[ch * szKernel];
                    for (size_t i = 0; i < szOutput; ++i) {
                        number sum = 0;
                        std::vector <size_t> &row = indexmap[i];
                        for (size_t k = 0; k < szKernel; ++k) {
                            if (row[k] == szInput) {
                                continue;
                            }
                            sum += cKernel[k].weight * cInput[row[k]];
                        }
                        cPooling[i] += sum;
                    }
                }

                for (size_t i = 0; i < szOutput; ++i) {
                    cPooling[i] += bias.weight; // apply bias
                }
            }
            //data[bz].displayOutput();
        }
    }
}

void ConvolutionalLayer::backprop(){
    for(size_t bz = 0; bz < this->size; bz++){
        calcBiasChanges(bz);
        calcKernelChanges(bz);
        if(calcLeftErrorSignal){
            calcInputChanges(bz);
        }
    }


}

void ConvolutionalLayer::calcKernelChanges(size_t bz){
    auto & rightErrorSignal = this->getRightErrorSignal(bz);
    auto & input = this->getInput(bz);

    auto & idxInput = this->getActiveInput(bz);
    auto & idxRightErrorSignal = this->getActiveRightErrorSignal(bz);

    if(idxRightErrorSignal.size() == 0){ // dense trace
        if(idxInput.size() == 0){ // dense input
            for (size_t och = 0; och < outChannel; och++) {
                number *cPoolingErrorSignal = &rightErrorSignal[och * szOutput];
                auto &kernel = filter[och].kernel;

                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &input[ch * szInput];
                    auto *cKernel = &kernel[ch * szKernel];

                    for (size_t i = 0; i < szOutput; i++) {
                        for (size_t k = 0; k < szKernel; ++k) {

                            if (indexmap[i][k] == szInput) {
                                continue;
                            }

                            size_t index = indexmap[i][k];
                            //sDebug() << k <<"+=" << index  <<"*"<< i;

                            number changes = cInput[index] * cPoolingErrorSignal[i];
                            cKernel[k].gradient += changes;
                        }
                    }
                }
            }
        } else { // sparse input

            std::vector<number> denseInput;
            denseInput.resize(this->leftActive->fullsize);
            for(size_t i = 0; i < idxInput.size(); i++){
                denseInput[idxInput[i]] = input[i];
            }

            for (size_t och = 0; och < outChannel; och++) {
                number *cPoolingErrorSignal = &rightErrorSignal[och * szOutput];
                auto &kernel = filter[och].kernel;

                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &denseInput[ch * szInput];
                    auto *cKernel = &kernel[ch * szKernel];

                    for (size_t i = 0; i < szOutput; i++) {
                        for (size_t k = 0; k < szKernel; ++k) {

                            if (indexmap[i][k] == szInput) {
                                continue;
                            }

                            size_t index = indexmap[i][k];
                            number changes = cInput[index] * cPoolingErrorSignal[i];
                            cKernel[k].gradient += changes;
                        }
                    }
                }
            }
        }
    } else { // sparse trace
        if(idxInput.size() == 0){
            for (size_t och = 0; och < outChannel; och++) {

                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &input[ch * szInput];
                    //auto *cKernel = &kernel[ch * szKernel];

                    for (size_t r = 0; r < rightErrorSignal.size(); r++) {
                        for (size_t k = 0; k < szKernel; ++k) {
                            size_t idxR = idxRightErrorSignal[r];
                            size_t idxC = idxR % szInput;
                            size_t orch = idxR / szInput;

                            if (indexmap[idxC][k] == szInput) {
                                continue;
                            }

                            size_t index = indexmap[idxC][k];
                            number changes = cInput[index] * rightErrorSignal[r];
                            filter[orch].kernel[ch * szKernel + k].gradient += changes;
                        }
                    }
                }
            }
        } else {
            qDebug() << "not implemented yet";
        }

    }
}

void ConvolutionalLayer::calcBiasChanges(size_t bz){
    if(desc.learnbias){

        auto &rightErrorSignal = nextLayer->getLeftErrorSignal(bz);
        auto &idxRightErrorSignal = this->getActiveRightErrorSignal(bz);

        if(idxRightErrorSignal.size() == 0){ // dense errorsignal
            for (size_t och = 0; och < outChannel; och++) {
                   auto &bias = filter[och].bias;
                   number *cPoolingErrorSignal = &rightErrorSignal[och * szOutput];

                   for (size_t i = 0; i < szOutput; i++) {
                       bias.gradient += cPoolingErrorSignal[i];
                   }
               }
         } else {
            for (size_t r = 0; r < idxRightErrorSignal.size(); r++){
                size_t och = idxRightErrorSignal[r] / szOutput;
                auto &bias = filter[och].bias;
                bias.gradient += rightErrorSignal[r];

            }
        }
    }
}

void ConvolutionalLayer::calcInputChanges(size_t bz){

    auto &leftErrorSignal = this->getLeftErrorSignal(bz);
    auto &idxLeftErrorSignal = this->getActiveLeftErrorSignal(bz);
    auto &rightErrorSignal = this->getRightErrorSignal(bz);
    auto &idxRightErrorSignal = this->getActiveRightErrorSignal(bz);

    leftErrorSignal.resize(szInput*inChannel);
    std::fill(&leftErrorSignal[0], &leftErrorSignal[szInput * inChannel], 0);

    if(idxRightErrorSignal.size() == 0){
       //sDebug() << "calc input error signal";
        for (size_t och = 0; och < outChannel; och++) {
            number *cPoolingErrorSignal = &rightErrorSignal[och * szOutput];
            for (size_t ch = 0; ch < inChannel; ch++) {

                number *cInputErrorSignal = &leftErrorSignal[ch * szInput];
                auto *cKernel = &filter[och].kernel[ch * szKernel];

                for (size_t i = 0; i < szOutput; ++i) {
                    size_t traceidx = i % szInput;
                    for (size_t k = 0; k < szKernel; ++k) {
                        if (indexmap[traceidx][k] == szInput) {
                            continue;
                        }
                        number changes = cKernel[k].weight * cPoolingErrorSignal[i];
                        cInputErrorSignal[indexmap[traceidx][k]] += changes;
                    }
                }
            }
        }
    } else {

            for (size_t ch = 0; ch < inChannel; ch++) {
                number *cInputErrorSignal = &leftErrorSignal[ch * szInput];

                for(size_t r = 0; r < idxRightErrorSignal.size(); r++){ // todo swap och (r) and ch ?
                    size_t idxR = idxRightErrorSignal[r];
                    size_t idxC = idxR % szInput;
                    size_t orch = idxR / szInput;

                    for (size_t k = 0; k < szKernel; ++k) {
                        if (indexmap[idxC][k] == szInput) {
                            continue;
                        }
                        number changes = filter[orch].kernel[ch * szKernel + k].weight * rightErrorSignal[r];
                        cInputErrorSignal[indexmap[idxC][k]] += changes;
                    }
                }
            //}
        }
}
}

void ConvolutionalLayer::setRandom()
{
    number scaling = 1.0 /sqrt( (outChannel * szKernel));
    std::uniform_real_distribution<number> dist(-scaling, scaling);

    for(size_t och =0; och < outChannel; ++och){
        for(size_t ch = 0; ch < inChannel; ++ch){
            for(size_t i = 0; i < szKernel; i++){
                this->filter[och].kernel[i+ch*szKernel].weight = dist(RandomDevice::engine);
                this->filter[och].kernel[i+ch*szKernel].gradient = 0;
            }
        }
    }
}

void ConvolutionalLayer::createIndexmap(ConvolutionLayerDescription desc)
{
    indexmap.resize(szOutput);
    for (size_t i = 0; i < szOutput; ++i) {
        auto &row = indexmap[i];
        row.resize(szKernel);
        row.shrink_to_fit();
    }

    auto scatter = desc.scatter;
    auto offset = desc.offset;
    auto leftPadding = desc.leftPadding;
    auto rightPadding = desc.rightPadding;
    auto lower = desc.lower;
    auto upper = desc.upper;

    std::vector<size_t> inputcoords = desc.lower;
    std::vector<size_t> kernelcoords;
    std::vector<size_t> targetcoords = dimInput.zeroCoords();

    for (size_t i = 0; i < szOutput; i++) {
        kernelcoords = dimKernel.zeroCoords();
        for (size_t k = 0; k < szKernel; k++) {
            size_t index; // avoid error for goto
            for (size_t ik = 0; ik < dimInput.dim; ik++) {
                int delta;

                delta = int(inputcoords[ik]) + int(scatter[ik]) * (-int(kernelcoords[ik]) + offset[ik]);

                if (delta < 0) {

                    if (leftPadding[ik] == PaddingType::Zerofill) {
                        indexmap[i][k] = szInput;
                        goto skip;
                    } else if (leftPadding[ik] == PaddingType::Same) {
                        delta = 0;
                    } else if (leftPadding[ik] == PaddingType::Torus) {
                        delta = int(dimInput.gridsize[ik]) + delta;
                    }

                } else if (delta >= dimInput.gridsize[ik]) {

                    if (rightPadding[ik] == PaddingType::Zerofill) {
                        indexmap[i][k] = szInput;
                        goto skip;
                    } else if (rightPadding[ik] == PaddingType::Same) {
                        delta = int(dimInput.gridsize[ik]) - 1;
                    } else if (rightPadding[ik] == PaddingType::Torus) {
                        delta = delta - int(dimInput.gridsize[ik]);
                    }
                }
                targetcoords[ik] = size_t(delta);
            }
            index = dimInput.index(&targetcoords[0]);
            indexmap[i][k] = index;

            skip: // todo replace by bool flag?
            dimKernel.inc(&kernelcoords[0]);
        }
        dimInput.inc(&inputcoords[0], &lower[0], &upper[0]);
    }

}

void ConvolutionalLayer::displayIndexmap(){

}

void ConvolutionalLayer::displayKernel()
{
    sDebug() << "============ Kernel ================";
    std::stringstream ss;
    for (size_t och = 0; och < outChannel; ++och) {
        //ss << "output channel: " << och + 1 << "/" << outChannel << "\n";

        for (size_t ch = 0; ch < inChannel; ++ch){
            ss << "Kernel_" << och + 1<<"_"<<ch+1 << " = [";
            sDebug() << ss.str().c_str();
            ss = std::stringstream();
        auto *kernel = &filter[och].kernel[ch*szKernel];

        for (size_t i = 0; i < szKernel; i++) {
            ss << kernel[i].weight << ", ";

            size_t d = dimKernel.dim -1;
            size_t div = dimKernel.gridsize[d];
            while((i+1) % div == 0 && d != 0){
                div = div * dimKernel.gridsize[d-1];
                sDebug() << ss.str().c_str();
                ss = std::stringstream();
                d--;
            }
        }
        sDebug() << ss.str().c_str()<<"]";
        }
    }
}

void ConvolutionalLayer::displayKernelChanges()
{

}

void ConvolutionalLayer::displayBias()
{

}

void ConvolutionalLayer::displayBiasChanges()
{

}

void ConvolutionalLayer::update(size_t epoch){
    for (size_t och = 0; och < outChannel; och++) {
       auto &bias = filter[och].bias;
       auto &kernel = filter[och].kernel;

       for (size_t k = 0; k < szKernel * inChannel; ++k) {
           filter[och].kernelUpdater.update(kernel[k], epoch);
       }

       filter[och].biasUpdater.update(bias, epoch); // todo move updater?
   }
}

void ConvolutionalLayer::serialize(std::ostream &out){
    // todo
}
