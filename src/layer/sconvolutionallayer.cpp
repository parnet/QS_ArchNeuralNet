#include "sconvolutionallayer.h"
#include "randomdevice.h"

SConvolutionalLayer::SConvolutionalLayer(): AbstractLayer() {
    this->type = LayerType::Convolution;
    sDebug() << "Neither description nor previous layer provided for SConvolutional Layer";
}

SConvolutionalLayer::SConvolutionalLayer(AbstractLayer *prev): AbstractLayer(prev) {
    this->type = LayerType::Convolution;
    sDebug() << "No description provided for SConvolutional Layer";
}

SConvolutionalLayer::SConvolutionalLayer(ConvolutionLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev), dimInput(desc.dimInput),
    dimOutput(desc.dimOutput), dimKernel(desc.dimKernel) {

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
    this->createDualmap();

    // todo learningrate kernel
    // todo learningrate bias
    // todo inputlayer calc



}

SConvolutionalLayer::SConvolutionalLayer(std::istream &stream, AbstractLayer *prev):AbstractLayer(stream, prev) {
    // todo
}

SConvolutionalLayer::~SConvolutionalLayer()
{

}

void SConvolutionalLayer::init(){

}

void SConvolutionalLayer::prepare(){
    this->size = previousLayer->size;
    this->data.clear();
    this->data.resize(this->size, GeneralData());
}

void SConvolutionalLayer::feedforward(){
    for(size_t bz = 0; bz < this->size; bz++){
        auto input = this->getInput(bz);
        auto idxInput = this->getActiveInput(bz);

        auto & output = this->getOutput(bz);
        output.resize(szOutput*outChannel);
        std::fill(&output[0], &output[szOutput*outChannel], 0.0 );

        auto idxOutput = this->getActiveOutput(bz);
        idxOutput.clear();

        if(idxInput.size() != 0){
            // make it dense
            //std::vector<number> denseInput;
            //denseInput.resize(this->leftActive->fullsize);
            //for(size_t i = 0; i < idxInput.size(); i++){
            //    denseInput[idxInput[i]] = input[i];
            //}

            for (size_t och = 0; och < outChannel; och++) {
                auto &kernel = filter[och].kernel;
                auto &bias = filter[och].bias;
                number *cPooling = &output[och * szOutput];

                size_t idx = 0;
                for (size_t ch = 0; ch < inChannel; ch++) {
                    //number *cInput = &input[ch * szInput];
                    auto *cKernel = &kernel[ch * szKernel];

                    std::vector<number> partialOutput;
                    partialOutput.resize(szOutput,0);

                    for (; idx < idxInput.size(); ++idx) {
                        if(idxInput[idx] >= (ch+1)*szInput){
                            break;
                        }
                        size_t idxCh = idxInput[idx] % szInput;
                        if(idxCh + ch*szInput != idxInput[idx]){
                            sDebug() << "inconsistent index " << idxCh << "in"<<ch<<" basing on " << idxInput[idx];
                        }


                        auto &row = dualmap[idxCh];
                        for (size_t k = 0; k < szKernel; ++k) {
                            if (row[k] == szOutput) {
                                continue;
                            }
                            if(row[k] >= szOutput){
                                sDebug() << "index too big in dualmap";
                            }
                            partialOutput[row[k]] += cKernel[k].weight * input[idx];
                        }
                    }

                    for(size_t pp = 0; pp < szOutput; pp++){
                        cPooling[pp] += partialOutput[pp];
                    }
                }

                for (size_t i = 0; i < szOutput; ++i) {
                    cPooling[i] += bias.weight; // apply bias
                }
            }

        } else { // input is dense use normal convolution method as in ConvolutionLayer
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

        }
        //displayOutput(bz);
    }
}

void SConvolutionalLayer::backprop()
{
    for(size_t bz = 0; bz < this->size; bz++){
        calcBiasChanges(bz);
        calcKernelChanges(bz);
        if(calcLeftErrorSignal){
            calcInputChanges(bz);
        }
    }
}

void SConvolutionalLayer::update(size_t epoch) {
    for (size_t och = 0; och < outChannel; och++) {
       auto &bias = filter[och].bias;
       auto &kernel = filter[och].kernel;

       for (size_t k = 0; k < szKernel * inChannel; ++k) {
           filter[och].kernelUpdater.update(kernel[k], epoch);
       }

       filter[och].biasUpdater.update(bias, epoch); // todo move updater?
   }
}

void SConvolutionalLayer::serialize(std::ostream &out)
{

}

void SConvolutionalLayer::calcKernelChanges(size_t bz){

    auto & rightErrorSignal = this->getRightErrorSignal(bz);
    auto & idxRightErrorSignal = this->getActiveRightErrorSignal(bz);
    //auto & trace = this->getActiveRightErrorSignal(bz);

    auto & input = this->getInput(bz);
    auto & idxInput = this->getActiveInput(bz);



    if(idxRightErrorSignal.size() == 0){
        if(idxInput.size() == 0){ // dense input, dense output, dense right error signal -> use ConvolutionLayer backprop
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
        } else { // sparse input, dense output, dense right error signal
            for (size_t och = 0; och < outChannel; och++) {
                number *crightErrorSignal = &rightErrorSignal[och * szOutput];
                auto &kernel = filter[och].kernel;

                size_t idx = 0;
                for (size_t ch = 0; ch < inChannel; ch++) {
                    auto *cKernel = &kernel[ch * szKernel];

                    for(; idx < idxInput.size(); idx++){
                        if(idxInput[idx] >= (ch+1)*szInput){
                            break;
                        }
                        size_t chIdx = idxInput[idx] % szInput;

                        //assert(chIdx + ch*szInput == idxInput[idx] && "index missmatch");

                        for (size_t k = 0; k < szKernel; ++k) {
                            size_t index = dualmap[chIdx][k];
                            if (index == szOutput) {
                                continue;
                            }
                            if (index > szOutput) {
                                sDebug() << "index too big";
                            }


                            number changes = input[idx] * crightErrorSignal[index];
                            cKernel[k].gradient += changes;
                        }
                    }
                }
            }
        }
    } else { // sparse trace
        if(idxInput.size() == 0){ // dense input, dense output, sparse error signal
            //for (size_t och = 0; och < outChannel; och++) {
                //auto &kernel = filter[och].kernel;
                //size_t *cTrace = &idxRightErrorSignal[och * szOutput];

                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &input[ch * szInput];

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
            //}
        } else { // sparse input, dense output, sparse error signal::  decide for dualmap or indexmap method?
            //qDebug() << "not implemented yet" << idxInput.size() * this->outChannel << "\t " << idxRightErrorSignal.size();
            // todo test
            std::vector<number> denseinput;
            denseinput.resize(inChannel*szInput);

            for(size_t j = 0; j < idxInput.size(); j++){
                denseinput[idxInput[j]] = input[j];
            }

            //for (size_t och = 0; och < outChannel; och++) {


                for (size_t ch = 0; ch < inChannel; ch++) {
                    number *cInput = &denseinput[ch * szInput];

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
            //}
            /*

            for (size_t och = 0; och < outChannel; och++) {
                //number *crightErrorSignal = &rightErrorSignal[och * szOutput];
                auto &kernel = filter[och].kernel;

                size_t idx = 0;
                for (size_t ch = 0; ch < inChannel; ch++) {
                    auto *cKernel = &kernel[ch * szKernel];

                    for(; idx < idxInput.size(); idx++){
                        if(idxInput[idx] >= (ch+1)*szInput){
                            break;
                        }
                        size_t chIdx = idxInput[idx] % szInput;

                        assert(chIdx + ch*szInput == idxInput[idx] && "index missmatch");

                        for (size_t k = 0; k < szKernel; ++k) {
                            size_t index = dualmap[chIdx][k];

                            if (index == szOutput) {
                                continue;
                            }

                            if (index > szOutput) {
                                sDebug() << "index too big";
                            }


                            number changes = input[idx] * crightErrorSignal[index];
                            cKernel[k].gradient += changes;
                        }
                    }
                }
            }





*/














        }

    }
}

void SConvolutionalLayer::calcBiasChanges(size_t bz){
    if(desc.learnbias){
    auto &rightErrorSignal = nextLayer->getLeftErrorSignal(bz);
    for (size_t och = 0; och < outChannel; och++) {
               auto &bias = filter[och].bias;
               number *cPoolingErrorSignal = &rightErrorSignal[och * szOutput];

               for (size_t i = 0; i < szOutput; i++) {
                   bias.gradient += cPoolingErrorSignal[i];
               }
           }
    }
}

void SConvolutionalLayer::calcInputChanges(size_t bz){

    auto &leftErrorSignal = this->getLeftErrorSignal(bz);
    auto &idxLeftErrorSignal = this->getActiveLeftErrorSignal(bz);
    auto &rightErrorSignal = this->getRightErrorSignal(bz);
    auto &idxRightErrorSignal = this->getActiveRightErrorSignal(bz);

    leftErrorSignal.resize(szInput*inChannel);
    std::fill(&leftErrorSignal[0], &leftErrorSignal[szInput * inChannel], 0);

    if(idxRightErrorSignal.size() == 0){ // dense right error signal
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
    } else { // sparse right Error Signal

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

    }
    }
}

void SConvolutionalLayer::setRandom(){
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
    this->displayKernel();
}

void SConvolutionalLayer::createIndexmap(ConvolutionLayerDescription desc){
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
            0;
        }
        dimInput.inc(&inputcoords[0], &lower[0], &upper[0]);
    }
}

void SConvolutionalLayer::createDualmap(){
    /*
     * indexmap : output x kernel -> input
     * dualmap  : input x kernel -> output
     */
    dualmap.resize(szInput);
    for(size_t inp = 0; inp < szInput; inp++){
        dualmap[inp].resize(szKernel, szOutput);
        dualmap[inp].shrink_to_fit();
    }

    for(size_t out = 0; out < szOutput; out++){
        for(size_t k = 0; k < szKernel; k++){
            size_t ind = indexmap[out][k];
            if(ind == szInput){
                continue;
            }
            dualmap[ind][k] = out;
        }
    }
}

void SConvolutionalLayer::displayIndexmap(){
    int fill = int(ceil(log10(szInput)));
    sDebug() << "============ Indexmap ================";
    for (size_t i = 0; i < szOutput; i++) {
        std::stringstream ss;
        ss << std::setfill('0');
        for (size_t k = 0; k < szKernel; k++) {
            ss << std::setw(fill)<< indexmap[i][k] << ", ";
        }
        sDebug() << ss.str().c_str();
    }
    sDebug() << "\n";
}

void SConvolutionalLayer::displayDualmap(){
    // todo fill to match inputsize
    int fill = int(ceil(log10(szOutput)));
    sDebug() << "============ Dualmap ================";
    for (size_t i = 0; i < szInput; i++) {
        std::stringstream ss;
        ss << std::setfill('0');
        for (size_t k = 0; k < szKernel; k++) {
            ss << std::setw(fill)<< dualmap[i][k] << ", ";
        }
        sDebug() << ss.str().c_str();
    }
    sDebug() << "\n";

}

void SConvolutionalLayer::displayKernel() {
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

void SConvolutionalLayer::displayKernelChanges(){
    sDebug() << "============ Kernel Weight Changes ================";
    std::stringstream ss;
    for (size_t och = 0; och < outChannel; ++och) {
        ss << "output channel: " << och + 1 << "/" << outChannel << "\n";

        for (size_t ch = 0; ch < inChannel; ++ch){
        auto *kernel = &filter[och].kernel[ch*szKernel];

        std::stringstream ss;
        for (size_t i = 0; i < szKernel; i++) {
            ss << kernel[i].gradient << ", ";

            size_t d = dimKernel.dim -1;
            size_t div = dimKernel.gridsize[d];
            while((i+1) % div == 0 && d != 0){
                div = div * dimKernel.gridsize[d-1];
                sDebug() << ss.str().c_str();
                ss = std::stringstream();
                d--;
            }
        }
        sDebug() << ss.str().c_str();
    }
}
}

void SConvolutionalLayer::displayBias(){
    sDebug() << "============ Bias  ================";
    for (size_t och = 0; och < outChannel; och++) {
        sDebug() << "Channel " << och << ": " << filter[och].bias.weight;
    }
}

void SConvolutionalLayer::displayBiasChanges(){
    sDebug() << "============ Bias Changes  ================";
    for (size_t och = 0; och < outChannel; och++) {
        sDebug() << "Channel " << och << ": " << filter[och].bias.gradient;
    }
}
/*

void SConvolutionalLayer::displayInput(size_t index) {
    auto &input = getInput(index);
    sDebug() << "==================== Input ====================";
    for (size_t ch = 0; ch < inChannel; ch++) {
        number *cInput = &input[ch * szInput];
        // sDebug() << "Input Channel" << ch;
        sDebug() << "Input_" << ch << "= [";
        std::stringstream ss;
        for (size_t i = 0; i < szInput; i++) {
            ss << cInput[i] << ", ";

            size_t d = dimOutput.dim -1; // todo check dim Output ? dim Input
            size_t div = dimOutput.gridsize[d];
            while((i+1) % div == 0 && d != 0){
                div = div * dimOutput.gridsize[d-1];
                sDebug() << ss.str().c_str();
                ss = std::stringstream();
                d--;
            }
        }
        sDebug() << ss.str().c_str() << "]\n";
    }
}

void SConvolutionalLayer::displayOutput(size_t index) {
    sDebug() << "=============== Convolution Output ===============";
       for (size_t och = 0; och < outChannel; och++) {
           sDebug() << "Output Channel" << och;
           number *cPooling = &this->data[index].output[och * szOutput];
           std::stringstream ss;
           for (size_t i = 0; i < szOutput; i++) {
               ss << cPooling[i] << ", ";

               size_t d = dimOutput.dim -1;
               size_t div = dimOutput.gridsize[d];
               while((i+1) % div == 0 && d != 0){
                   div = div * dimOutput.gridsize[d-1];
                   sDebug() << ss.str().c_str();
                   ss = std::stringstream();
                   d--;
               }
           }
           sDebug() << ss.str().c_str() << "\n";
       }
}

void SConvolutionalLayer::displayRightErrorSignal(size_t index){
    // todo?
}

void SConvolutionalLayer::displayLeftErrorSignal(size_t index){
    sDebug() << "============ input error signal ================";
    for (size_t ch = 0; ch < inChannel; ch++) {
        number *cInputErrorSigna = &this->data[index].errorSignal[ch * szInput];
        sDebug() << "Input Channel " << ch;
        std::stringstream ss;
        for (size_t i = 0; i < szInput; i++) {
            ss << cInputErrorSigna[i] << ", ";

            size_t d = dimOutput.dim -1;
            size_t div = dimOutput.gridsize[d];
            while((i+1) % div == 0 && d != 0){
                div = div * dimOutput.gridsize[d-1];
                sDebug() << ss.str().c_str();
                ss = std::stringstream();
                d--;
            }
        }
        sDebug() << ss.str().c_str();
    }
}
*/
