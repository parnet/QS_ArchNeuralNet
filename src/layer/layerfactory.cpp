#include "activationlayer.h"
#include "averagepoolinglayer.h"
#include "combinedlayer.h"
#include "convolutionallayer.h"
#include "euklidianpoolinglayer.h"
#include "fullyconnectedlayer.h"
#include "layerfactory.h"
#include "lppoolinglayer.h"
#include "maxpoolinglayer.h"
#include "normalizationlayer.h"
#include "sconvolutionallayer.h"
#include "selectiveconvolutionlayer.h"
#include "sparseconnectedlayer.h"
#include "stochasticpoolinglayer.h"

LayerFactory::LayerFactory(){
    
}

AbstractLayer *LayerFactory::createLayer(VariantLayerDescriptions desc, AbstractLayer *prev){
    AbstractLayer * rval = nullptr;
    auto index = desc.index();
    switch(index){
        case 1: rval = new ActivationLayer(std::get<1>(desc), prev); break;
        case 2: rval = new InputLayer(std::get<2>(desc), prev); break;
        case 3: rval = new OutputLayer(std::get<3>(desc), prev); break;
        case 4: rval = new FullyConnectedLayer(std::get<4>(desc), prev); break;
        case 5: rval = new SConvolutionalLayer(std::get<5>(desc), prev); break;
        case 6: rval = new MaxPoolingLayer(std::get<6>(desc), prev); break;
    default:
        sDebug() << "layer not instanciable"; // todo
    }
    return rval;
}


AbstractLayer *LayerFactory::createLayer(AbstractLayerDescription *desc, AbstractLayer *prev){

    /*

    / general layer types /
    AbstractLayer * layerin   = new InputLayer();
    AbstractLayer * layerout  = new OutputLayer();
    AbstractLayer * layeract  = new ActivationLayer();

    / connection layer types /
    AbstractLayer * layerfc   = new FullyConnectedLayer();
    AbstractLayer * layersc   = new SparseConnectedLayer();

    / advanced layer types /
     AbstractLayer * layernorm = new NormalizationLayer();

     / convolution /
     AbstractLayer * layerconv = new ConvolutionalLayer();
     AbstractLayer * layersel  = new SelectiveConvolutionLayer();

     / pooling layer types /
     AbstractLayer * layermax  = new MaxPoolingLayer();
     AbstractLayer * layeravg  = new AveragePoolingLayer();
     AbstractLayer * layereuk  = new EuklidianPoolingLayer();
     AbstractLayer * layerlp   = new LpPoolingLayer();
     AbstractLayer * layersp   = new StochasticPoolingLayer();

     / composition /
     AbstractLayer * layercmb  = new CombinedLayer();*/



    //return rval;
    return nullptr;
}

AbstractLayer *LayerFactory::createLayer(LayerType type, std::istream stream, AbstractLayer *prev){
    // todo

    /* general layer types */
    AbstractLayer * layerin   = new InputLayer(stream, prev);
    AbstractLayer * layerout  = new OutputLayer(stream, prev);
    AbstractLayer * layeract  = new ActivationLayer(stream, prev);

    /* connection layer types */
    AbstractLayer * layerfc   = new FullyConnectedLayer(stream, prev);
    // AbstractLayer * layersc   = new SparseConnectedLayer(stream, prev);

    /* advanced layer types */
     // AbstractLayer * layernorm = new NormalizationLayer(stream, prev);

     /* convolution */
     // AbstractLayer * layerconv = new ConvolutionalLayer(stream, prev);
     // AbstractLayer * layersel  = new SelectiveConvolutionLayer(stream, prev);

     /* pooling layer types */
     // AbstractLayer * layermax  = new MaxPoolingLayer(stream, prev);
     // AbstractLayer * layeravg  = new AveragePoolingLayer(stream, prev);
     // AbstractLayer * layereuk  = new EuklidianPoolingLayer(stream, prev);
     // AbstractLayer * layerlp   = new LpPoolingLayer(stream, prev);
     // AbstractLayer * layersp   = new StochasticPoolingLayer(stream, prev);

     /* composition */
     // AbstractLayer * layercmb  = new CombinedLayer(stream, prev);

}
