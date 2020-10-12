#ifndef LAYERFACTORY_H
#define LAYERFACTORY_H


#include <variant>

#include "abstractlayer.h"
#include "abstractlayerdescription.h"
#include "inputlayer.h"
#include "outputlayer.h"


#include <activationlayerdescription.h>
#include <convolutionlayerdescription.h>
#include <fullyconnecteddescription.h>
#include <maxpoolinglayerdescription.h>


typedef std::variant<AbstractLayerDescription,
                     ActivationLayerDescription,
                     InputLayerDescription,
                     OutputLayerDescription,
                     FullyConnectedDescription,
                     ConvolutionLayerDescription,
                     MaxPoolingLayerDescription>
    VariantLayerDescriptions;

class LayerFactory{
public:
    LayerFactory();

    static AbstractLayer * createLayer(VariantLayerDescriptions desc, AbstractLayer * prev);

    static AbstractLayer * createLayer(AbstractLayerDescription * desc, AbstractLayer * prev);

    static AbstractLayer * createLayer(LayerType type, std::istream stream, AbstractLayer * prev);


};

#endif // LAYERFACTORY_H
