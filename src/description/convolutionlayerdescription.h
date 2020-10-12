#ifndef CONVOLUTIONLAYERDESCRIPTION_H
#define CONVOLUTIONLAYERDESCRIPTION_H

#include "abstractlayerdescription.h"
#include "paddingtype.h"


class ConvolutionLayerDescription :public AbstractLayerDescription {
public:
    ConvolutionLayerDescription() : AbstractLayerDescription(LayerType::Convolution){

    }

    size_t inChannel;
    size_t outChannel;
    std::vector<size_t> dimInput;
    std::vector<size_t> dimOutput;
    std::vector<size_t> dimKernel;
    std::vector<size_t> stride;

    std::vector<PaddingType> leftPadding;
    std::vector<PaddingType> rightPadding;
    std::vector<size_t> scatter;
    std::vector<size_t> offset;

    std::vector<size_t> lower;
    std::vector<size_t> upper;

    bool learnbias = false;
    bool learnkernel = true;


};

#endif // CONVOLUTIONLAYERDESCRIPTION_H
