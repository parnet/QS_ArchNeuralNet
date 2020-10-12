#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "environment.h"
#include <activationlayerdescription.h>
#include <convolutionlayerdescription.h>
#include <fullyconnecteddescription.h>
#include <inputlayerdescription.h>
#include <maxpoolinglayerdescription.h>
#include <outputlayerdescription.h>

#include "layerfactory.h"

class Topology{

public:

    std::vector<VariantLayerDescriptions> layers;
public:
    Topology();

    void addDescription(VariantLayerDescriptions desc);

    ~Topology();

    static Topology buildConv();

    static Topology defaultTopology();

    static Topology experimentalTopology();

    static Topology defaultNoneHidden();

    static Topology defaultSingleHidden();

    static Topology defaultDoubleHidden();

    static Topology defaultConvolutional();

    static Topology getPredefined(size_t index);
};

#endif // TOPOLOGY_H
