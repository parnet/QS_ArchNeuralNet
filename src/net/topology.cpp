#include "topology.h"

#include <activationlayerdescription.h>
#include <convolutionlayerdescription.h>
#include <inputlayerdescription.h>

Topology::Topology(){

}


void Topology::addDescription(VariantLayerDescriptions desc){
    this->layers.push_back(desc);
}


Topology::~Topology(){}

Topology Topology::buildConv(){
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,24};

    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput); // todo check Dimension(4,20,20,20,28)
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);


    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {5,3,5};
    descConv.scatter = {1,1,1};

    descConv.offset = {(descConv.dimKernel[0] -1)/2,(descConv.dimKernel[1] -1)/2,(descConv.dimKernel[2] -1)};
    descConv.leftPadding = {PaddingType::Torus, PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Torus, PaddingType::Zerofill, PaddingType::Zerofill};

    descConv.inChannel = 28;
    descConv.outChannel = 4;
    descConv.dimInput = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM};
    descConv.lower = {0,0,0};
    descConv.upper = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM};
    descConv.dimOutput = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM};
    top.addDescription(descConv);

    /*ConvolutionLayerDescription descConv;
    descConv.dimKernel = {5,3,5};
    descConv.scatter = {1,1,1};
    descConv.offset = {descConv.dimKernel[0]-1, (descConv.dimKernel[1] -1)/2,(descConv.dimKernel[2] -1)/2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Torus};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Torus};
    descConv.inChannel = 28;
    descConv.outChannel = 4;
    descConv.dimInput = {GS_MOMENTUM, GS_POLAR, GS_INCLINATION};
    descConv.lower = {0,0,0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    top.addDescription(descConv);*/


    ActivationLayerDescription descActivationConv;
    descActivationConv.dropout = 0.0;
    descActivationConv.usesbias = false;
    descActivationConv.activation = NeuronType::LeakyReLU;
    descActivationConv.numberOfNeurons = descConv.outChannel*descConv.dimOutput[2]*descConv.dimOutput[0]*descConv.dimOutput[1];
    top.addDescription(descActivationConv);


    MaxPoolingLayerDescription descPooling;
    descPooling.channel = descConv.outChannel;
    descPooling.dimInput = descConv.dimOutput;
    descPooling.stride = {4,1,2};
    descPooling.dimOutput = {descPooling.dimInput[0] / descPooling.stride[0],
                             descPooling.dimInput[1] / descPooling.stride[1],
                            descPooling.dimInput[2] / descPooling.stride[2]};
    top.addDescription(descPooling);


    FullyConnectedDescription descConnection;
    descConnection.szLeft = 224000;
    descConnection.szRight = 2;
    top.addDescription(descConnection);

    ActivationLayerDescription descActivation;
    descActivation.dropout = 0.0;
    descActivation.usesbias = false;
    descActivation.activation = NeuronType::Softmax;
    descActivation.numberOfNeurons = 2;
    top.addDescription(descActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);

    return top;
}

Topology Topology::defaultTopology(){
    return Topology::buildConv();
}

Topology Topology::experimentalTopology()
{
    // todo
    return Topology::defaultTopology();
    Topology top;

    InputLayerDescription descInput;
    top.addDescription(descInput);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3,3,3};
    descConv.offset = {(descConv.dimKernel[1] -1)/2, (descConv.dimKernel[1] -1)/2,(descConv.dimKernel[2] -1)/2};
    descConv.inChannel = 28;
    descConv.outChannel = 4;
    descConv.dimInput = {GS_MOMENTUM, GS_POLAR, GS_INCLINATION};
    descConv.lower = {0,0,0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    top.addDescription(descConv);

    ActivationLayerDescription descConvActivation;
    descConvActivation.usesbias = false;
    descConvActivation.numberOfNeurons = descConv.outChannel * descConv.dimOutput.size();
    descConvActivation.activation = NeuronType::ReLU;
    top.addDescription(descConvActivation);

    MaxPoolingLayerDescription descPooling;
    top.addDescription(descPooling);

    FullyConnectedDescription descConnection;
    top.addDescription(descConnection);

    ActivationLayerDescription descOutputActivation;
    top.addDescription(descOutputActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);

    return top;
}

Topology Topology::defaultNoneHidden(){
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,28};
    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput); // todo check Dimension(4,20,20,20,28)
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descConnection;
    descConnection.szLeft = 224000;
    descConnection.szRight = 2;
    top.addDescription(descConnection);

    ActivationLayerDescription descActivation;
    descActivation.dropout = 0.0;
    descActivation.usesbias = false;
    descActivation.activation = NeuronType::Softmax;
    descActivation.numberOfNeurons = 2;
    top.addDescription(descActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);

    return top;
}

Topology Topology::defaultSingleHidden(){
    Topology top;

    InputLayerDescription descInput;
    std::vector<size_t> dimInput = {20,20,20,28};
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descHiddenConnection;
    descHiddenConnection.szLeft = 224000;
    descHiddenConnection.szRight = 64;
    top.addDescription(descHiddenConnection);

    ActivationLayerDescription descHiddenActivation;
    descHiddenActivation.dropout = 0.5;
    descHiddenActivation.usesbias = true;
    descHiddenActivation.activation = NeuronType::LeakyReLU;
    descHiddenActivation.numberOfNeurons = 64;
    top.addDescription(descHiddenActivation);


    FullyConnectedDescription descConnection;
    descConnection.szLeft = 64;
    descConnection.szRight = 2;
    top.addDescription(descConnection);

    ActivationLayerDescription descActivation;
    descActivation.dropout = 0.0;
    descActivation.usesbias = false;
    descActivation.activation = NeuronType::Softmax;
    descActivation.numberOfNeurons = 2;
    top.addDescription(descActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);


    return top;
}

Topology Topology::defaultDoubleHidden(){
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,28};
    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descHiddenConnectionA;
    descHiddenConnectionA.szLeft = 224000;
    descHiddenConnectionA.szRight = 8;
    top.addDescription(descHiddenConnectionA);

    ActivationLayerDescription descHiddenActivationA;
    descHiddenActivationA.dropout = 0.5;
    descHiddenActivationA.usesbias = true;
    descHiddenActivationA.activation = NeuronType::LeakyReLU;
    descHiddenActivationA.numberOfNeurons = 8;
    top.addDescription(descHiddenActivationA);

    FullyConnectedDescription descHiddenConnectionB;
    descHiddenConnectionB.szLeft = 8;
    descHiddenConnectionB.szRight = 8;
    top.addDescription(descHiddenConnectionB);

    ActivationLayerDescription descHiddenActivationB;
    descHiddenActivationB.dropout = 0.3;
    descHiddenActivationB.usesbias = false;
    descHiddenActivationB.activation = NeuronType::LeakyReLU;
    descHiddenActivationB.numberOfNeurons = 8;
    top.addDescription(descHiddenActivationA);

    FullyConnectedDescription descConnection;
    descConnection.szLeft = 8;
    descConnection.szRight = 2;
    top.addDescription(descConnection);

    ActivationLayerDescription descActivation;
    descActivation.dropout = 0.0;
    descActivation.usesbias = false;
    descActivation.activation = NeuronType::Softmax;
    descActivation.numberOfNeurons = 2;
    top.addDescription(descActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);
    return top;
}

Topology Topology::defaultConvolutional(){
    Topology top;

    InputLayerDescription descInput;
    std::vector<size_t> dimInput = {20,20,20,28};
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    ConvolutionLayerDescription descConvA;{
    descConvA.inChannel = 28;
    descConvA.outChannel = 32;

    descConvA.dimInput = {20,20,20};

    descConvA.dimKernel = {3,3,3};

    descConvA.offset = {(descConvA.dimKernel[0]-1)/2,
                        (descConvA.dimKernel[1]-1)/2,
                        (descConvA.dimKernel[2]-1)/2 };

    descConvA.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvA.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvA.scatter = {1,1,1};

    descConvA.lower = {0,0,0};

    descConvA.upper = {descConvA.dimInput[0],
                       descConvA.dimInput[1],
                       descConvA.dimInput[2]};

    descConvA.dimOutput = {descConvA.upper[0] - descConvA.lower[0],
                           descConvA.upper[1] - descConvA.lower[1],
                           descConvA.upper[2] - descConvA.lower[2]};

    descConvA.learnkernel = true;

    descConvA.learnbias = true;

    top.addDescription(descConvA);}

    ActivationLayerDescription descConvActivationA;
    top.addDescription(descConvActivationA);

    MaxPoolingLayerDescription descPoolingA;
    top.addDescription(descPoolingA);

    ConvolutionLayerDescription descConvB;{
    descConvB.inChannel = 32;
    descConvB.outChannel = 64;

    descConvB.dimInput = {10,10,10};

    descConvB.dimKernel = {3,3,3};

    descConvB.offset = {(descConvB.dimKernel[0]-1)/2,
                        (descConvB.dimKernel[1]-1)/2,
                        (descConvB.dimKernel[2]-1)/2 };

    descConvB.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvB.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvB.scatter = {1,1,1};

    descConvB.lower = {0,0,0};

    descConvB.upper = {descConvB.dimInput[0],
                       descConvB.dimInput[1],
                       descConvB.dimInput[2]};

    descConvB.dimOutput = {descConvB.upper[0] - descConvB.lower[0],
                           descConvB.upper[1] - descConvB.lower[1],
                           descConvB.upper[2] - descConvB.lower[2]};

    descConvB.learnkernel = true;

    descConvB.learnbias = true;

    top.addDescription(descConvB);}

    ActivationLayerDescription descConvActivationB;
    top.addDescription(descConvActivationB);

    MaxPoolingLayerDescription descPoolingB;
    top.addDescription(descPoolingB);

    FullyConnectedDescription descConnection;{
    descConnection.szLeft = 8000;
    descConnection.szRight = 64;
    top.addDescription(descConnection);

    ActivationLayerDescription descHiddenActivation;
    descHiddenActivation.dropout = 0.0;
    descHiddenActivation.usesbias = false;
    descHiddenActivation.activation = NeuronType::ReLU;
    descHiddenActivation.numberOfNeurons = 64;
    top.addDescription(descHiddenActivation);}

    FullyConnectedDescription descConnectionL;{
    descConnectionL.szLeft = 64;
    descConnectionL.szRight = 2;
    top.addDescription(descConnectionL);

    ActivationLayerDescription descOutputActivation;
    descOutputActivation.dropout = 0.0;
    descOutputActivation.usesbias = false;
    descOutputActivation.activation = NeuronType::Softmax;
    descOutputActivation.numberOfNeurons = 2;
    top.addDescription(descOutputActivation);}


    OutputLayerDescription descOutput;
    top.addDescription(descOutput);

    return top;
}

Topology Topology::getPredefined(size_t index)
{
    Topology descTopology;
#ifndef EXPERIMENTAL
    index = index + 1;
#endif
    switch (index) {
        case 0:
            descTopology = Topology::experimentalTopology();
              break;
        case 1:
            descTopology = Topology::defaultTopology();
            break;
        case 2:
            descTopology = Topology::defaultSingleHidden();
            break;
        case 3:
            descTopology = Topology::defaultDoubleHidden();
            break;
        case 4: // CNN
            descTopology = Topology::defaultConvolutional();
            break;
        default:
            descTopology = Topology::defaultTopology();
        break;
}
    return descTopology;
}
