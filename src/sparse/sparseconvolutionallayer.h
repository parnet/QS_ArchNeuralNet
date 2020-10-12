#ifndef SPARSECONVOLUTIONALLAYER_H
#define SPARSECONVOLUTIONALLAYER_H

#include "sparseabstractlayer.h"
#include "sparseconvolutionaldriver.h"
#include "sparsedata.h"

#include <convolutionlayerdescription.h>


/*
class SparseConvolutionalLayer : public SparseAbstractLayer {
public:
    std::vector<SparseData> data;
    SparseConvolutionalDriver driver;

    ConvolutionLayerDescription desc;
public:
    SparseConvolutionalLayer();

    SparseConvolutionalLayer(AbstractLayer * prev);

    SparseConvolutionalLayer(ConvolutionLayerDescription desc, AbstractLayer * prev);

    SparseConvolutionalLayer(std::istream & stream, AbstractLayer * prev);

    ~SparseConvolutionalLayer() override;
};*/

#endif // SPARSECONVOLUTIONALLAYER_H
