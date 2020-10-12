#ifndef FULLYCONNECTEDDESCRIPTION_H
#define FULLYCONNECTEDDESCRIPTION_H

#include "abstractlayerdescription.h"
#include "environment.h"

class FullyConnectedDescription : public AbstractLayerDescription
{
public:
    FullyConnectedDescription() : AbstractLayerDescription(LayerType::FullyConnected){}
    size_t szLeft;
    size_t szRight;


};

#endif // FULLYCONNECTEDDESCRIPTION_H
