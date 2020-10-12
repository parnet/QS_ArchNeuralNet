#ifndef ACTIVATIONLAYERDESCRIPTION_H
#define ACTIVATIONLAYERDESCRIPTION_H

#include "abstractlayerdescription.h"


class ActivationLayerDescription : public AbstractLayerDescription {
public:
    ActivationLayerDescription(): AbstractLayerDescription(LayerType::Activation){}

    bool usesbias = false;
    number dropout = 0;

    NeuronType activation;
    size_t numberOfNeurons;
};




#endif // ACTIVATIONLAYERDESCRIPTION_H
