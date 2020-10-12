#ifndef OUTPUTLAYERDESCRIPTION_H
#define OUTPUTLAYERDESCRIPTION_H

#include "abstractlayerdescription.h"



class OutputLayerDescription : public AbstractLayerDescription
{
public:
    OutputLayerDescription() : AbstractLayerDescription(LayerType::Output){}

};

#endif // OUTPUTLAYERDESCRIPTION_H
