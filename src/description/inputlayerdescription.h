#ifndef INPUTLAYERDESCRIPTION_H
#define INPUTLAYERDESCRIPTION_H

#include "abstractlayerdescription.h"

#include <dimension.h>

class InputLayerDescription : public AbstractLayerDescription
{
public:
    InputLayerDescription() : AbstractLayerDescription(LayerType::Input){}

    Dimension dimension;
    size_t size;

};

#endif // INPUTLAYERDESCRIPTION_H
