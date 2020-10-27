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


Topology Topology::AGI_Topology(){ // Artemy, Grigory, Ivan
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,28};

    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput); // todo check Dimension(4,20,20,20,28)
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);


    ConvolutionLayerDescription descConvFirst;
    descConvFirst.dimKernel = {3,3,3};
    descConvFirst.scatter = {1,1,1};

    descConvFirst.offset = {(descConvFirst.dimKernel[0] -1)/2,
                       (descConvFirst.dimKernel[1] -1)/2,
                       (descConvFirst.dimKernel[2] -1)/2};

    descConvFirst.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};
    descConvFirst.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvFirst.inChannel = 28;
    descConvFirst.outChannel = 32;
    descConvFirst.dimInput = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM};

    descConvFirst.lower = {0,0,0};
    descConvFirst.upper = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM};

    descConvFirst.learnbias = true;
    descConvFirst.learnkernel = true;

    descConvFirst.dimOutput = {GS_INCLINATION,GS_POLAR, GS_MOMENTUM}; // depending size
    top.addDescription(descConvFirst);


    ActivationLayerDescription descActivationConv;
    descActivationConv.dropout = 0.0; // todo dropout ?
    descActivationConv.usesbias = false;
    descActivationConv.activation = NeuronType::LeakyReLU;
    descActivationConv.numberOfNeurons = descConvFirst.outChannel*descConvFirst.dimOutput[2]*descConvFirst.dimOutput[0]*descConvFirst.dimOutput[1];
    top.addDescription(descActivationConv);


    MaxPoolingLayerDescription descPooling;
    descPooling.channel = descConvFirst.outChannel;
    descPooling.dimInput = descConvFirst.dimOutput;
    descPooling.stride = {2,2,2};
    descPooling.dimOutput = {descPooling.dimInput[0] / descPooling.stride[0],
                             descPooling.dimInput[1] / descPooling.stride[1],
                            descPooling.dimInput[2] / descPooling.stride[2]};
    top.addDescription(descPooling);




    ConvolutionLayerDescription descConvSecond;
    descConvSecond.dimKernel = {3,3,3};
    descConvSecond.scatter = {1,1,1};

    descConvSecond.offset = {(descConvSecond.dimKernel[0] -1)/2,
                       (descConvSecond.dimKernel[1] -1)/2,
                       (descConvSecond.dimKernel[2] -1)/2};

    descConvSecond.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};
    descConvSecond.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Zerofill};

    descConvSecond.inChannel = 32;
    descConvSecond.outChannel = 64;
    descConvSecond.dimInput = descPooling.dimOutput;

    descConvSecond.lower = {0,0,0};
    descConvSecond.upper = descConvSecond.dimInput;
    descConvSecond.learnbias = true;

    descConvSecond.dimOutput = descConvSecond.dimInput; // depnding size
    top.addDescription(descConvSecond);


    ActivationLayerDescription descActivationConvSecond;
    descActivationConvSecond.dropout = 0.0;
    descActivationConvSecond.usesbias = false;
    descActivationConvSecond.activation = NeuronType::LeakyReLU;
    descActivationConvSecond.numberOfNeurons = descConvSecond.outChannel*descConvSecond.dimOutput[2]*descConvSecond.dimOutput[0]*descConvSecond.dimOutput[1];
    top.addDescription(descActivationConvSecond);


    MaxPoolingLayerDescription descPoolingSecond;
    descPoolingSecond.channel = descConvSecond.outChannel;
    descPoolingSecond.dimInput = descConvSecond.dimOutput;
    descPoolingSecond.stride = {2,2,2};
    descPoolingSecond.dimOutput = {descPoolingSecond.dimInput[0] / descPoolingSecond.stride[0],
                             descPoolingSecond.dimInput[1] / descPoolingSecond.stride[1],
                            descPoolingSecond.dimInput[2] / descPoolingSecond.stride[2]};
    top.addDescription(descPoolingSecond);


    FullyConnectedDescription descConnection;
    descConnection.szLeft = 8000;
    descConnection.szRight = 64;
    top.addDescription(descConnection);

    ActivationLayerDescription descActivation;
    descActivation.dropout = 0.0; // todo dropout of 0.5
    descActivation.usesbias = true;
    descActivation.activation = NeuronType::LeakyReLU;
    descActivation.numberOfNeurons = 64;
    top.addDescription(descActivation);

    FullyConnectedDescription descConnectionLast;
    descConnectionLast.szLeft = 64;
    descConnectionLast.szRight = 2;
    top.addDescription(descConnectionLast);

    ActivationLayerDescription descActivationLast;
    descActivationLast.dropout = 0.0;
    descActivationLast.usesbias = false;
    descActivationLast.activation = NeuronType::Softmax;
    descActivationLast.numberOfNeurons = 2;
    top.addDescription(descActivationLast);


    OutputLayerDescription descOutput;
    top.addDescription(descOutput);

    return top;
}

// LinearSeperable() 224000 -> 2   (80%)
// SingleHiddenLayer_64() 224000 -> 64 -> 2
// DoubleHiddenLayer_8_8() 224000 -> 8 -> 8 -> 2




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
    return Topology::linearSeperable();
}

Topology Topology::experimentalTopology()
{
    return Topology::defaultNoneHidden();
    //return Topology::defaultTopology();
    Topology top;

    InputLayerDescription descInput;
    top.addDescription(descInput);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3,3,5};
    descConv.offset = {(descConv.dimKernel[1] -1)/2, (descConv.dimKernel[1] -1)/2,(descConv.dimKernel[2] -1)/2};
    descConv.inChannel = 28;
    descConv.outChannel = 8;
    descConv.dimInput = {GS_MOMENTUM, GS_POLAR, GS_INCLINATION};
    descConv.lower = {0,0,0};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Torus};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill, PaddingType::Torus};
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

Topology Topology::linearSeperable(){
    Topology top;

    InputLayerDescription descInput;
    std::vector<size_t> dimInput = {20,20,20,28}; // todo replace by dimension
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descHiddenConnection;
    descHiddenConnection.szLeft = 224000;
    descHiddenConnection.szRight = 2;
    top.addDescription(descHiddenConnection);

    ActivationLayerDescription descOutputActivation;
    descOutputActivation.dropout = 0.0;
    descOutputActivation.usesbias = false;
    descOutputActivation.activation = NeuronType::Softmax;
    descOutputActivation.numberOfNeurons = 2;
    top.addDescription(descOutputActivation);

    OutputLayerDescription descOutput;
    top.addDescription(descOutput);


    return top;
}

Topology Topology::singleHidden_64()
{
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,28};
    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descHiddenConnectionA;
    descHiddenConnectionA.szLeft = 224000;
    descHiddenConnectionA.szRight = 64;
    top.addDescription(descHiddenConnectionA);

    ActivationLayerDescription descHiddenActivationA;
    descHiddenActivationA.dropout = 0.5;
    descHiddenActivationA.usesbias = true;
    descHiddenActivationA.activation = NeuronType::LeakyReLU;
    descHiddenActivationA.numberOfNeurons = descHiddenConnectionA.szRight;
    top.addDescription(descHiddenActivationA);

    FullyConnectedDescription descConnection;
    descConnection.szLeft = descHiddenActivationA.numberOfNeurons;
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

Topology Topology::doubleHidden_64_64(){
    Topology top;

    std::vector<size_t> dimInput = {20,20,20,28};
    InputLayerDescription descInput;
    descInput.dimension = Dimension(dimInput);
    descInput.size = descInput.dimension.size();
    top.addDescription(descInput);

    FullyConnectedDescription descHiddenConnectionA;
    descHiddenConnectionA.szLeft = 224000;
    descHiddenConnectionA.szRight = 64;
    top.addDescription(descHiddenConnectionA);

    ActivationLayerDescription descHiddenActivationA;
    descHiddenActivationA.dropout = 0.5;
    descHiddenActivationA.usesbias = true;
    descHiddenActivationA.activation = NeuronType::LeakyReLU;
    descHiddenActivationA.numberOfNeurons = descHiddenConnectionA.szRight;
    top.addDescription(descHiddenActivationA);

    FullyConnectedDescription descHiddenConnectionB;
    descHiddenConnectionB.szLeft = descHiddenActivationA.numberOfNeurons;
    descHiddenConnectionB.szRight = 64;
    top.addDescription(descHiddenConnectionB);

    ActivationLayerDescription descHiddenActivationB;
    descHiddenActivationB.dropout = 0.5;
    descHiddenActivationB.usesbias = true;
    descHiddenActivationB.activation = NeuronType::LeakyReLU;
    descHiddenActivationB.numberOfNeurons = descHiddenConnectionB.szRight;
    top.addDescription(descHiddenActivationA);

    FullyConnectedDescription descConnection;
    descConnection.szLeft = descHiddenActivationB.numberOfNeurons;
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
    switch (index) {
        case 0:
            sDebug() << "0 fc expected linear seperable\n";
            descTopology = Topology::linearSeperable();
              break;
        case 1:
            sDebug() << "0 fc single singleHidden_64\n";
             descTopology = Topology::singleHidden_64();
            break;
        case 2:
            sDebug() << "0 fc doubleHidden_8_8\n";
            descTopology = Topology::doubleHidden_64_64();
            break;
        case 3:
            sDebug() << "0 fc AGI\n";
            descTopology = Topology::AGI_Topology();
            break;
        case 4: // CNN
            sDebug() << "0 fc expected\n";
            descTopology = Topology::defaultNoneHidden();
            break;
        default:
            sDebug() << "0 fc expected\n";
            descTopology = Topology::defaultNoneHidden();
        break;
}
    return descTopology;
}
