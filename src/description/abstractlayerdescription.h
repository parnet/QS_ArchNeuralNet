#ifndef ABSTRACTLAYERDESCRIPTION_H
#define ABSTRACTLAYERDESCRIPTION_H

#include "environment.h"
#include "dimension.h"
#include "layertype.h"
#include "neurontype.h"

class AbstractLayerDescription {
public:
    AbstractLayerDescription(LayerType type){ this->type = type;}

    ~AbstractLayerDescription() = default;

    LayerType type;
};


#endif // ABSTRACTLAYERDESCRIPTION_H
