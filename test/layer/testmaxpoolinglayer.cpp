#include "testmaxpoolinglayer.h"

#include <maxpoolinglayer.h>
#include <randomdevice.h>
#include <testlayer.h>

#include "testutil.h"

int __TestMaxPoolingLayer::all(){
    int failed = 0;
    failed += !cycle_dd_dd();
    failed += !cycle_dd_sd();
    sDebug() << "Test: Max Pooling Layer" << "Failed: " << failed;
    return failed;
}

bool __TestMaxPoolingLayer::cycle_dd_dd()
{
    RandomDevice::setSeed(1024);
    TestLayer * input = new TestLayer(nullptr);
    input->setRightVectorSize(64);
    input->setSize(1);
    input->prepare();

    MaxPoolingLayerDescription descMax;
    descMax.stride = {2,2};
    descMax.channel = 2;
    descMax.dimInput = {4,8};
    descMax.dimOutput = {2,4};

    MaxPoolingLayer * maxlayer = new MaxPoolingLayer(descMax, input);

    TestLayer * output = new TestLayer(maxlayer);
    output->setSize(1);
    output->prepare();

    output->setLeftVectorSize(32);

    maxlayer->init();
    maxlayer->prepare();

    std::vector<number> v = RandomDevice::createUniformVector(64,-1.0,1.0);

    input->setOutput(0, v , {});
    //input->displayOutput(0);
    //input->displayActiveOutput(0);

    output->setExpectedInput(0, {// channel 0
                                 0.764729995719024, -0.174004049440184, 0.765885360560244, 0.533941202467329,0.207324073611935,0.880199323104835, 0.770558773563769, 0.379463500401410,
                                 // channel 1
                                 0.820968714871392,0.378911753342968,0.956456841160527,0.643772817717563,0.694088967850016,0.215137651488749,-0.502873571270331,0.715102399063339
                             }, {});

    auto e = RandomDevice::createUniformVector(16,-1.0, 1.0);
    output->setErrorSignal(0, e, {});

    //output->displayLeftErrorSignal(0);
    //output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {-0.679029162453132,0.416188861890348,0.39704578587799,-0.93850957217091,-0.852409222272776,0.738950582237963,-0.322723431560683,0.414399288509083,
                                       0.154188546406671,-0.0875019411221668,-0.522202980094368,-0.677151717720451,0.102744292615251,-0.256767626241875,0.767662670017139,-0.306504437470121},
                                     {5,7,8,11,16,23,24,31,33,39,41,43,53,51,57,58});

    maxlayer->feedforward();

    //maxlayer->displayActiveLeftErrorSignal(0);

    bool chkFeedForward = output->controlInput();
    if(!chkFeedForward){
        sDebug() << "Feedforward not passed";
    }

    maxlayer->backprop();


    bool chkBackProp = input->controlErrorSignal();

    if(!chkBackProp){
        sDebug() << "Backpropagation not passed";
    }

    delete input;
    delete maxlayer;
    delete output;

    return chkFeedForward && chkBackProp;
}

bool __TestMaxPoolingLayer::cycle_dd_sd() {
    RandomDevice::setSeed(1024);
    TestLayer * input = new TestLayer(nullptr);
    input->setRightVectorSize(64);
    input->setSize(1);
    input->prepare();

    MaxPoolingLayerDescription descMax;
    descMax.stride = {2,2};
    descMax.channel = 2;
    descMax.dimInput = {4,8};
    descMax.dimOutput = {2,4};

    MaxPoolingLayer * maxlayer = new MaxPoolingLayer(descMax, input);

    TestLayer * output = new TestLayer(maxlayer);
    output->setSize(1);
    output->prepare();

    output->setLeftVectorSize(32);

    maxlayer->init();
    maxlayer->prepare();

    std::vector<number> v = RandomDevice::createUniformVector(64,-1.0,1.0);

    input->setOutput(0, v , {});
    //input->displayOutput(0);
    //input->displayActiveOutput(0);

    output->setExpectedInput(0, {// channel 0
                                 0.764729995719024, -0.174004049440184, 0.765885360560244, 0.533941202467329,0.207324073611935,0.880199323104835, 0.770558773563769, 0.379463500401410,
                                 // channel 1
                                 0.820968714871392,0.378911753342968,0.956456841160527,0.643772817717563,0.694088967850016,0.215137651488749,-0.502873571270331,0.715102399063339
                             }, {});

    auto e = RandomDevice::createUniformVector(16,-1.0, 1.0);
    auto mas = RandomDevice::createMask(16,0.5);
    auto errx = applyMask(e,mas);

    output->setErrorSignal(0, errx, mas);

    //output->displayLeftErrorSignal(0);
    //output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {-0.679029162453132,0.416188861890348,0.39704578587799,-0.93850957217091,-0.852409222272776,0.738950582237963,-0.322723431560683,0.414399288509083,
                                      0.154188546406671,-0.0875019411221668,-0.522202980094368,-0.677151717720451},
                                     {7,8,11,16,24,31,33,39,41,43,53,58});

    maxlayer->feedforward();

    bool chkFeedForward = output->controlInput();
    if(!chkFeedForward){
        sDebug() << "Feedforward not passed";
    }

    maxlayer->backprop();


    bool chkBackProp = input->controlErrorSignal();

    if(!chkBackProp){
        sDebug() << "Backpropagation not passed y";
    }

    delete input;
    delete maxlayer;
    delete output;

    return chkFeedForward && chkBackProp;
}
