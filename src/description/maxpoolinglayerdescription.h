#ifndef MAXPOOLINGLAYERDESCRIPTION_H
#define MAXPOOLINGLAYERDESCRIPTION_H

#include "abstractlayerdescription.h"
#include "environment.h"
class MaxPoolingLayerDescription : AbstractLayerDescription
{
public:
    MaxPoolingLayerDescription() : AbstractLayerDescription(LayerType::MaxPooling){}

    size_t channel;

    std::vector<size_t> dimInput;
    std::vector<size_t> stride;
    std::vector<size_t> dimOutput;

};

#endif // MAXPOOLINGLAYERDESCRIPTION_H
