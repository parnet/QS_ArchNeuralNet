#include "testconvolutionlayer.h"

#include <convolutionallayer.h>
#include <convolutionlayerdescription.h>
#include <randomdevice.h>
#include <testlayer.h>

#include "testutil.h"

bool __TestConvolutionLayer::checkBias(std::vector<number> expected, ConvolutionalLayer *clayer)
{
    number tol = 1e-5;
    bool passed = true;
    for(size_t i = 0; i < clayer->filter.size(); i++){
        if(fabs(expected[i] - clayer->filter[i].bias.gradient) > tol){
            sDebug() << "tolerance not given for filter " << i << " bias  exp: "<<expected[i]<< " given: "<<clayer->filter[i].bias.gradient;
            passed = false;
        }
    }
    return passed;
}

bool __TestConvolutionLayer::checkKernel(std::vector<std::vector<number> > expected, ConvolutionalLayer *clayer)
{
    number tol = 1e-5;
    bool passed = true;
    for(size_t i = 0; i < clayer->filter.size(); i++){
        for(size_t j = 0; j < clayer->filter.size(); j++){
            if(fabs(expected[i][j] - clayer->filter[i].kernel[j].gradient) > tol){
                sDebug() << "tolerance not given for filter " << i << " weight " << j << " exp: "<<expected[i][j]<< " given: "<<clayer->filter[i].kernel[j].gradient;
                passed = false;
            }
        }
    }
    return passed;
}

int __TestConvolutionLayer::all(){
    int failed = 0;
    failed += !cycle_dd_dd();
    failed += !cycle_dd_sd();

    sDebug() << "Test: Convolutional Layer" << " Failed: "<< failed;
    return failed;
}


bool __TestConvolutionLayer::prepare(){
    RandomDevice::setSeed(1024);
    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3,3};
    descConv.scatter = {1,1};
    descConv.offset = {(descConv.dimKernel[0] -1)/2,(descConv.dimKernel[1] -1)/2};

    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4,8};
    descConv.lower = {0,0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;




    TestLayer * input = new TestLayer(nullptr);
    input->setRightVectorSize(4*8);
    input->setSize(1);
    input->prepare();


    ConvolutionalLayer *clayer = new ConvolutionalLayer(descConv,input);

    TestLayer * output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();

    output->setLeftVectorSize(4*8);

    clayer->init();
//    clayer->displayBias();
//    clayer->displayKernel();
    clayer->prepare();




    input->setOutput(0, RandomDevice::createUniformVector(4*8,-1.0,1.0), {});
//    input->displayOutput(0);

    output->setExpectedInput(0, {  0.037460516783488, // basing on "formart short" of kernel weights
                                   -0.173609272960799,
                                   -0.084158969021322,
                                    0.123935492213060,
                                    0.136480025821629,
                                    0.327378290199055,
                                   -0.656514696484321,
                                    0.143482145339272,
                                    0.182777791516837,
                                    0.188096398297131,
                                    0.358536839233877,
                                   -0.119866421306721,
                                    0.037725489329821,
                                   -0.235662933032006,
                                   -0.383434639640720,
                                   -0.190386939368249,
                                    0.098242405357026,
                                    0.020463037197982,
                                    0.076648003842177,
                                    0.329059949383023,
                                    0.306480285898288,
                                   -0.741947294916665,
                                   -0.109907159665743,
                                    0.072552351348516,
                                   -0.211025547600271,
                                   -0.054346185964880,
                                    0.055446721745243,
                                   -0.235250755735330,
                                    0.001426904848334,
                                    0.292993740915649,
                                   -0.153027156951560,
                                    0.059850016321624}, {});

    output->setErrorSignal(0,                   { 0,
                           0.956456841160527,
                           0.304919842993824,
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                           0.643772817717563,
                                           0,
                                           0,
                           0.795893563181871,
                                           0,
                           0.347110087181130,
                           0.184542191796150,
                                           0,
                          -0.429764894889161,
                           0.413789598629343,
                          -0.271827301467713,
                                           0,
                                           0,
                                           0,
                          -0.705229313584006,
                           0.215137651488749,
                                           0,
                                           0,
                                           0,
                          -0.250947407299365,
                                           0,
                                           0,
                                           0,
                                           0}, {});
//    output->displayLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {  -0.134072470663191,  -0.273564021175372,   0.091023028122959,  -0.341579146029962,
                                        -0.161407504624675,  -0.087717719052266,   0.095869004112476,   0.162565929641302,
                                        -0.179601846984936,   0.273640802107291,  -0.217017889096874,   0.078392824953237,
                                         0.115499682902053,  -0.386593165999028,  -0.285304422372999,   0.336881913137017,
                                         0.614276639456587,   0.217913616872323,  -0.515674038018486,   0.006288595107376,
                                         0.200880720293756,  -0.272813392940889,  -0.381204551839135,   0.150611007379229,
                                        -0.318775686638431,  -0.415248337497482,   0.219862649416764,   0.363433861541848,
                                        -0.298628007182585,   0.292046344705694,   0.124797461254647,  -0.344575014864946}, {});
    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if(!chkFeedForward){
        sDebug() << "Feedforward not passed";
    }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-0.313847058372537,  -1.670764454180794,   1.273391410494602,
                                            -0.752995376187398,   0.894695806473894,   0.312758019057688,
                                            -1.158564193318736,  -0.239559243311760,  -1.671173049171142}
                                     }, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({0}, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Bias)";
    }

    bool chkBackProp = input->controlErrorSignal();

    if(!chkBackProp){
        sDebug() << "Backpropagation not passed";
    }

    delete input;
    delete clayer;
    delete output;

    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}



bool __TestConvolutionLayer::cycle_dd_dd(){
    RandomDevice::setSeed(1024);
    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3,3};
    descConv.scatter = {1,1};
    descConv.offset = {(descConv.dimKernel[0] -1)/2,(descConv.dimKernel[1] -1)/2};

    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4,8};
    descConv.lower = {0,0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;




    TestLayer * input = new TestLayer(nullptr);
    input->setRightVectorSize(4*8);
    input->setSize(1);
    input->prepare();


    ConvolutionalLayer *clayer = new ConvolutionalLayer(descConv,input);

    TestLayer * output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();

    output->setLeftVectorSize(4*8);

    clayer->init();
    //clayer->displayBias();
    //clayer->displayKernel();
    clayer->prepare();




    input->setOutput(0, RandomDevice::createUniformVector(4*8,-1.0,1.0), {});
    //input->displayOutput(0);

    output->setExpectedInput(0, {  0.037460516783488, // basing on "formart short" of kernel weights
                                   -0.173609272960799,
                                   -0.084158969021322,
                                    0.123935492213060,
                                    0.136480025821629,
                                    0.327378290199055,
                                   -0.656514696484321,
                                    0.143482145339272,
                                    0.182777791516837,
                                    0.188096398297131,
                                    0.358536839233877,
                                   -0.119866421306721,
                                    0.037725489329821,
                                   -0.235662933032006,
                                   -0.383434639640720,
                                   -0.190386939368249,
                                    0.098242405357026,
                                    0.020463037197982,
                                    0.076648003842177,
                                    0.329059949383023,
                                    0.306480285898288,
                                   -0.741947294916665,
                                   -0.109907159665743,
                                    0.072552351348516,
                                   -0.211025547600271,
                                   -0.054346185964880,
                                    0.055446721745243,
                                   -0.235250755735330,
                                    0.001426904848334,
                                    0.292993740915649,
                                   -0.153027156951560,
                                    0.059850016321624}, {});

    output->setErrorSignal(0, RandomDevice::createUniformVector(4*8,-1.0, 1.0), {});
    //output->displayLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {  -0.134072470663191,  -0.273564021175372,   0.091023028122959,  -0.341579146029962,
                                        -0.161407504624675,  -0.087717719052266,   0.095869004112476,   0.162565929641302,
                                        -0.179601846984936,   0.273640802107291,  -0.217017889096874,   0.078392824953237,
                                         0.115499682902053,  -0.386593165999028,  -0.285304422372999,   0.336881913137017,
                                         0.614276639456587,   0.217913616872323,  -0.515674038018486,   0.006288595107376,
                                         0.200880720293756,  -0.272813392940889,  -0.381204551839135,   0.150611007379229,
                                        -0.318775686638431,  -0.415248337497482,   0.219862649416764,   0.363433861541848,
                                        -0.298628007182585,   0.292046344705694,   0.124797461254647,  -0.344575014864946}, {});
    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if(!chkFeedForward){
        sDebug() << "Feedforward not passed";
    }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-0.313847058372537,  -1.670764454180794,   1.273391410494602,
                                            -0.752995376187398,   0.894695806473894,   0.312758019057688,
                                            -1.158564193318736,  -0.239559243311760,  -1.671173049171142}
                                     }, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({0}, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Bias)";
    }

    bool chkBackProp = input->controlErrorSignal();

    if(!chkBackProp){
        sDebug() << "Backpropagation not passed";
    }

    delete input;
    delete clayer;
    delete output;

    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}



bool __TestConvolutionLayer::cycle_dd_sd(){
    RandomDevice::setSeed(1024);
    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3,3};
    descConv.scatter = {1,1};
    descConv.offset = {(descConv.dimKernel[0] -1)/2,(descConv.dimKernel[1] -1)/2};

    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4,8};
    descConv.lower = {0,0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;




    TestLayer * input = new TestLayer(nullptr);
    input->setRightVectorSize(4*8);
    input->setSize(1);
    input->prepare();


    ConvolutionalLayer *clayer = new ConvolutionalLayer(descConv,input);

    TestLayer * output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();

    output->setLeftVectorSize(4*8);

    clayer->init();
    //clayer->displayBias();
    //clayer->displayKernel();
    clayer->prepare();




    input->setOutput(0, RandomDevice::createUniformVector(4*8,-1.0,1.0), {});
    //input->displayOutput(0);
    //input->displayActiveOutput(0);

    output->setExpectedInput(0, {  0.037460516783488, // basing on "formart short" of kernel weights
                                   -0.173609272960799,
                                   -0.084158969021322,
                                    0.123935492213060,
                                    0.136480025821629,
                                    0.327378290199055,
                                   -0.656514696484321,
                                    0.143482145339272,
                                    0.182777791516837,
                                    0.188096398297131,
                                    0.358536839233877,
                                   -0.119866421306721,
                                    0.037725489329821,
                                   -0.235662933032006,
                                   -0.383434639640720,
                                   -0.190386939368249,
                                    0.098242405357026,
                                    0.020463037197982,
                                    0.076648003842177,
                                    0.329059949383023,
                                    0.306480285898288,
                                   -0.741947294916665,
                                   -0.109907159665743,
                                    0.072552351348516,
                                   -0.211025547600271,
                                   -0.054346185964880,
                                    0.055446721745243,
                                   -0.235250755735330,
                                    0.001426904848334,
                                    0.292993740915649,
                                   -0.153027156951560,
                                    0.059850016321624}, {});

    auto dYfull = RandomDevice::createUniformVector(4*8,-1.0, 1.0);
    auto dYmask = RandomDevice::createMask(4*8,0.25);

    auto dY = applyMask(dYfull,dYmask);
    output->setErrorSignal(0, dY,dYmask);

    //output->displayLeftErrorSignal(0);
    //output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {    0.243810413380230,  -0.161052333218170,  -0.294533027852938,  -0.069629360907169,
                                          -0.219572425821756,   0.000354081106964,   0.359397331754330,  -0.006585180465111,
                                          -0.072102414236283,  -0.120027591689279,   0.169462934744601,  -0.210771121445741,
                                           0.286744388696080,  -0.021327617398275,  -0.288287116061466,   0.059341177446496,
                                           0.146635374795151,  -0.253157069106777,   0.133659124566965,   0.119696981272814,
                                          -0.124031705489110,  -0.140248341643220,   0.191959723523605,   0.086605022357070,
                                                           0,   0.134366635887847,  -0.179118764123734,  -0.006264398965374,
                                                           0,                   0,   0.047812758565341,  -0.026388876509379}, {});
    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if(!chkFeedForward){
        sDebug() << "Feedforward not passed";
    }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-0.640263847530830,   0.780496392090783,  -0.390356070842587,
                                              0.396031589634415,  -1.265206027381001,  -0.332673689508541,
                                             -0.236902276874364,  -1.234725146969383,  -0.375427767631146}
                                     }, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({0}, clayer);
    if(!chkKernelChanges){
        sDebug() << "Gradient calculation not passed (Bias)";
    }

    bool chkBackProp = input->controlErrorSignal();

    if(!chkBackProp){
        sDebug() << "Backpropagation not passed";
    }

    delete input;
    delete clayer;
    delete output;

    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}
