#include "convolutionaldriver.h"
#include "randomdevice.h"

ConvolutionalDriver::ConvolutionalDriver(){
    // todo
}

ConvolutionalDriver::ConvolutionalDriver(ConvolutionLayerDescription desc)
    : dimInput(desc.dimInput),
       dimOutput(desc.dimOutput), dimKernel(desc.dimKernel){
       if(!desc.learnbias){ // todo set for every filter
           //this->learningRateBias = 0.0;
       }

       if(!desc.learnkernel){ // todo set for every filter
           //this->learningRateKernel = 0.0;
       }

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
       // todo qDebug() << szKernel<<"Ä" << desc.predefkernel;
       //if(desc.predefkernel){
       //    this->setPredefinedKernel();
       //} else {
           this->setRandom(); //todo set predef
       //}
       this->desc = desc;
       this->createIndexmap(desc);
       //this->displayIndexmap();
   }

ConvolutionalDriver::ConvolutionalDriver(std::istream &file){
//todo
/*
    file >> this->inChannel;
    file >> this->outChannel;

    this->dimKernel = Dimension(file);
    this->dimInput = Dimension(file);
    this->dimOutput = Dimension(file);

    szInput = dimInput.size();

    szOutput = dimOutput.size();
    szKernel = dimKernel.size();

    ConvolutionLayerDescription desc;
    desc.inChannel = this->inChannel;
    desc.outChannel = this->outChannel;
    desc.dimKernel = this->dimKernel.gridsize;
    desc.dimInput = this->dimInput.gridsize;
    desc.dimOutput = this->dimOutput.gridsize;

    int tmp;

    desc.leftPadding.resize(dimInput.dim);
    for(size_t i = 0; i < dimInput.dim; i++){
        file >> tmp;
        desc.leftPadding[i] = static_cast<PaddingType>(tmp);
    }

    desc.rightPadding.resize(dimInput.dim);
    for(size_t i = 0; i < dimInput.dim; i++){
        file >> tmp;
        desc.rightPadding[i] = static_cast<PaddingType>(tmp);
    }
    desc.scatter.resize(dimInput.dim);
    for(size_t i = 0; i < dimInput.dim; i++){
        file >> tmp;
        desc.scatter[i] = tmp;
    }
    desc.offset.resize(dimKernel.dim);
    for(size_t i = 0; i < dimKernel.dim; i++){
        file >> tmp;
        desc.offset[i] = tmp;
    }
    desc.lower.resize(dimInput.dim);

    for(size_t i = 0; i < dimInput.dim; i++){
        file >> tmp;
        desc.lower[i] = tmp;
    }

    desc.upper.resize(dimInput.dim);
    for(size_t i = 0; i < dimInput.dim; i++){
        file >> tmp;
        desc.upper[i] = tmp;
    }

    file >> tmp;
    desc.learnbias = tmp;
    if(tmp == 0){ // todo for all filter
        //this->learningRateBias = 0.0;
    }

    file >> tmp;
    desc.learnkernel = tmp;
    if(tmp == 0){ // todo for all filter
        //this->learningRateKernel = 0.0;
    }


    filter.resize(outChannel);
    for (size_t och = 0; och < outChannel; och++) {
        file >> filter[och].bias.weight;
         filter[och].bias.gradient = 0;
        filter[och].kernel.resize(szKernel * inChannel);

            for(size_t k = 0; k < inChannel*szKernel; ++k){
                   file >>  filter[och].kernel[k].weight;
                }
        }
    this->desc = desc;
    this->createIndexmap(desc);*/
    }
void ConvolutionalDriver::setRandom(){
    // todo
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

void ConvolutionalDriver::createIndexmap(ConvolutionLayerDescription desc){
    // todo
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

   std::vector <size_t> inputcoords = desc.lower;
   std::vector <size_t> kernelcoords;
   std::vector <size_t> targetcoords = dimInput.zeroCoords();

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
           dimKernel.inccoordLeast(&kernelcoords[0]);
       }
       dimInput.inccoordLeast(&inputcoords[0], &lower[0], &upper[0]);
   }
   //this->displayIndexmap();
}


void ConvolutionalDriver::update(size_t epoch){
     for (size_t och = 0; och < outChannel; och++) {
        auto &bias = filter[och].bias;
        auto &kernel = filter[och].kernel;

        for (size_t k = 0; k < szKernel * inChannel; ++k) {
            filter[och].kernelUpdater.update(kernel[k], epoch);
        }

        filter[och].biasUpdater.update(bias, epoch); // todo move updater?
    }
}

void ConvolutionalDriver::displayIndexmap(){
    // todo fill to match inputsize
    int fill = int(ceil(log10(szOutput)));
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

void ConvolutionalDriver::displayKernel(){
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

void ConvolutionalDriver::displayKernelChanges(){
    // todo
    /*sDebug() << "============ Kernel Weight Changes ================";
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
    }*/
}

void ConvolutionalDriver::displayBias(){
    // todo
    /*sDebug() << "============ Bias  ================";
    for (size_t och = 0; och < outChannel; och++) {
        sDebug() << "Channel " << och << ": " << filter[och].bias.weight;
    }*/
}

void ConvolutionalDriver::displayBiasChanges(){
    // todo
    /*sDebug() << "============ Bias Changes  ================";
    for (size_t och = 0; och < outChannel; och++) {
        sDebug() << "Channel " << och << ": " << filter[och].bias.gradient;
    }*/
}

void ConvolutionalDriver::setPredefinedKernel() {
    // todo
    /*std::vector<number> kernel = {1.0, 2.0, 1.0,
                                  2.0, 4.0, 2.0,
                                  1.0, 2.0, 1.0
                                 };
    qDebug() << "Kernel" << szKernel;
    if(szKernel == 9){

    for(size_t och =0; och < outChannel; ++och){
        for(size_t ch = 0; ch < inChannel; ++ch){
            for(size_t i = 0; i < szKernel; i++){
                this->filter[och].kernel[i+ch*szKernel].weight = kernel[i] * 1.0/16.0;
                this->filter[och].kernel[i+ch*szKernel].gradient = 0;
            }
        }
    }
    }  else {
        this->setRandom();
    }

    displayKernel();*/
}

void ConvolutionalDriver::serialize(std::ostream &out){
    // todo
    /*const char * sendl = "\n";
    // channel
    out << inChannel << " ";
    out << outChannel << sendl;

    dimKernel.serialize(out);
    dimInput.serialize(out);
    dimOutput.serialize(out);

    for(size_t i = 0; i < desc.leftPadding.size();i++){
        out << desc.leftPadding[i] << " ";
    }
    out << sendl;

    for(size_t i = 0; i < desc.rightPadding.size();i++){
        out << desc.rightPadding[i] << " ";
    }
    out << sendl;

    for(size_t i = 0; i < desc.scatter.size();i++){
        out << desc.scatter[i] << " ";
    }
    out << sendl;

    for(size_t i = 0; i < desc.offset.size();i++){
        out << desc.offset[i] << " ";
    }
    out << sendl;

    for(size_t i = 0; i < desc.lower.size();i++){
        out << desc.lower[i] << " ";
    }
    out << sendl;

    for(size_t i = 0; i < desc.upper.size();i++){
        out << desc.upper[i] << " ";
    }
    out << sendl;
    out << desc.learnbias << " " << desc.learnkernel << sendl;

    size_t szFilter = this->filter.size();
    for(auto & v : this->filter){
        out << v.bias.weight << " ";
        for(auto & val : v.kernel){
            out << val.weight << " ";
        }
        out << sendl;
    }*/
}
