#include "testsconvolutionlayer.h"

#include <convolutionlayerdescription.h>
#include <randomdevice.h>
#include <testlayer.h>
#include "testutil.h"


bool __TestSConvolutionLayer::checkBias(std::vector <number> expected, SConvolutionalLayer *clayer) {
    number tol = 1e-5;
    bool passed = true;
    for (size_t i = 0; i < clayer->filter.size(); i++) {
        if (fabs(expected[i] - clayer->filter[i].bias.gradient) > tol) {
            sDebug() << "tolerance not given for filter " << i << " bias  exp: "<<expected[i]<< " given: "<<clayer->filter[i].bias.gradient;
            passed = false;
        }
    }
    return passed;
}

bool __TestSConvolutionLayer::checkKernel(std::vector <std::vector<number>> expected, SConvolutionalLayer *clayer) {
    number tol = 1e-5;
    bool passed = true;
    for (size_t i = 0; i < clayer->filter.size(); i++) {
        for (size_t j = 0; j < clayer->filter.size(); j++) {
            if (fabs(expected[i][j] - clayer->filter[i].kernel[j].gradient) > tol) {
                // sDebug() << "tolerance not given for filter " << i << " weight " << j << " exp: "<<expected[i][j]<< " given: "<<clayer->filter[i].kernel[j].gradient;
                passed = false;
            }
        }
    }
    return passed;
}

// number of input channel, feedforward ( type of input, type of output) backpropagation (type of errorsignal, type of left errorsignal) number of output channel

int __TestSConvolutionLayer::all() {
    sDebug() << "Test: SparseInput Convolution Layer";
    int failed = 0;
    failed += !cycle_1_dd_dd_1(); // C
    failed += !cycle_1_dd_sd_1(); // C

    failed += !cycle_1_ds_dd_1(); // C
    failed += !cycle_1_ds_sd_1(); // C

    failed += !cycle_1_dd_dd_2(); // C
    failed += !cycle_1_dd_sd_2(); // C

    failed += !cycle_1_ds_dd_2(); // C
    failed += !cycle_1_ds_sd_2(); // C

    failed += !cycle_2_dd_dd_1(); // C
    failed += !cycle_2_ds_dd_1(); // C

    failed += !cycle_2_ds_sd_1(); // O
    failed += !cycle_2_dd_sd_1(); // O

    failed += !cycle_2_ds_dd_2(); // C
    failed += !cycle_2_ds_sd_2(); // C

    failed += !cycle_2_dd_dd_2(); // C
    failed += !cycle_2_dd_sd_2(); // C
    if (failed != 0) {
        exit(-1);
    }
    return failed;
}

// -------------------------------------------------------------------- 1 in channel ----------------------------------------------
//           ---------------------------------------------------------        1 out channel ---------------------------------------
// checked group
bool __TestSConvolutionLayer::cycle_1_dd_dd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8);

    clayer->init();
    clayer->prepare();

    input->setOutput(0, RandomDevice::createUniformVector(4 * 8 * 1, -1.0, 1.0), {});
    output->setExpectedInput(0, {0.037460516783488, -0.173609272960799, -0.084158969021322, 0.123935492213060,
                                 0.136480025821629, 0.327378290199055, -0.656514696484321, 0.143482145339272,
                                 0.182777791516837, 0.188096398297131, 0.358536839233877, -0.119866421306721,
                                 0.037725489329821, -0.235662933032006, -0.383434639640720, -0.190386939368249,
                                 0.098242405357026, 0.020463037197982, 0.076648003842177, 0.329059949383023,
                                 0.306480285898288, -0.741947294916665, -0.109907159665743, 0.072552351348516,
                                 -0.211025547600271, -0.054346185964880, 0.055446721745243, -0.235250755735330,
                                 0.001426904848334, 0.292993740915649, -0.153027156951560, 0.059850016321624}, {});

    output->setErrorSignal(0, RandomDevice::createUniformVector(4 * 8, -1.0, 1.0), {});
    input->setExpectedErrorSignal(0, {-0.134072470663191, -0.273564021175372, 0.091023028122959, -0.341579146029962,
                                      -0.161407504624675, -0.087717719052266, 0.095869004112476, 0.162565929641302,
                                      -0.179601846984936, 0.273640802107291, -0.217017889096874, 0.078392824953237,
                                      0.115499682902053, -0.386593165999028, -0.285304422372999, 0.336881913137017,
                                      0.614276639456587, 0.217913616872323, -0.515674038018486, 0.006288595107376,
                                      0.200880720293756, -0.272813392940889, -0.381204551839135, 0.150611007379229,
                                      -0.318775686638431, -0.415248337497482, 0.219862649416764, 0.363433861541848,
                                      -0.298628007182585, 0.292046344705694, 0.124797461254647, -0.344575014864946},
                                  {});

    clayer->prepare();
    clayer->feedforward();
    //output->displayLeftErrorSignal(0);

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();

    bool chkKernelChanges = checkKernel({{-0.313847058372537, -1.670764454180794, 1.273391410494602,
                                                 -0.752995376187398, 0.894695806473894, 0.312758019057688,
                                                 -1.158564193318736, -0.239559243311760, -1.671173049171142}
                                        }, clayer);
    if (!chkKernelChanges) {
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({1.051284418948903}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }
    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_1_dd_sd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;
    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8);

    clayer->init();
    clayer->prepare();

    input->setOutput(0, RandomDevice::createUniformVector(4 * 8 * 1, -1.0, 1.0), {});
    output->setExpectedInput(0, {0.037460516783488, -0.173609272960799, -0.084158969021322, 0.123935492213060,
                                 0.136480025821629, 0.327378290199055, -0.656514696484321, 0.143482145339272,
                                 0.182777791516837, 0.188096398297131, 0.358536839233877, -0.119866421306721,
                                 0.037725489329821, -0.235662933032006, -0.383434639640720, -0.190386939368249,
                                 0.098242405357026, 0.020463037197982, 0.076648003842177, 0.329059949383023,
                                 0.306480285898288, -0.741947294916665, -0.109907159665743, 0.072552351348516,
                                 -0.211025547600271, -0.054346185964880, 0.055446721745243, -0.235250755735330,
                                 0.001426904848334, 0.292993740915649, -0.153027156951560, 0.059850016321624}, {});

    auto dYfull = RandomDevice::createUniformVector(4 * 8, -1.0, 1.0);
    auto dYmask = RandomDevice::createMask(4 * 8, 0.25);
    auto dY = applyMask(dYfull, dYmask);
    output->setErrorSignal(0, dY, dYmask);
    input->setExpectedErrorSignal(0, {0.243810413380230, -0.161052333218170, -0.294533027852938, -0.069629360907169,
                                      -0.219572425821756, 0.000354081106964, 0.359397331754330, -0.006585180465111,
                                      -0.072102414236283, -0.120027591689279, 0.169462934744601, -0.210771121445741,
                                      0.286744388696080, -0.021327617398275, -0.288287116061466, 0.059341177446496,
                                      0.146635374795151, -0.253157069106777, 0.133659124566965, 0.119696981272814,
                                      -0.124031705489110, -0.140248341643220, 0.191959723523605, 0.086605022357070,
                                      0, 0.134366635887847, -0.179118764123734, -0.006264398965374,
                                      0, 0, 0.047812758565341, -0.026388876509379}, {});

    clayer->prepare();
    clayer->feedforward();

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-0.640263847530830, 0.780496392090783, -0.390356070842587,
                                                 0.396031589634415, -1.265206027381001, -0.332673689508541,
                                                 -0.236902276874364, -1.234725146969383, -0.375427767631146}
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({2.203853676908912}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_1_ds_dd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8);

    clayer->init();
    clayer->prepare();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * 1, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * 1, 0.5);
    auto X = applyMask(Xflat, Xmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * 1);
    input->setOutput(0, X, Xmask);

    clayer->displayKernel();
    clayer->displayBias();
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {0, -0.013900942245441, -0.011262058319560, 0.020405191376393,
                                 0.017831183075847, 0.173864808523127, -0.055507110190687, -0.011512335946001,
                                 0.029611513653725, -0.352116463053354, -0.038547279395444, 0.116465123607043,
                                 -0.036364058124206, -0.032264677540081, 0.327628309612056, -0.330882176499300,
                                 -0.053018889624282, 0.078152735120199, 0.189887158897095, -0.133330767354904,
                                 -0.044148836678201, -0.009932782567500, 0.136359281535031, 0.368393325548477,
                                 -0.025545574072185, 0.186499859747414, 0.291057876039340, -0.110491886534209,
                                 0.031812249547410, 0.083008766926389, 0.015562944671884, -0.180775283868216}, {});

    output->setErrorSignal(0, RandomDevice::createUniformVector(4 * 8, -1.0, 1.0), {});
    input->setExpectedErrorSignal(0, {0.099603395407262, -0.128235254761723, 0.468991388617611, 0.106560670991232,
                                      0.298051679308613, -0.114118392576251, -0.211422898474380, -0.141324259833750,
                                      -0.338219277494206, 0.126613027815615, -0.101410376969625, 0.307478388190980,
                                      -0.562380100174939, -0.452447294127136, 0.428851691952449, 0.234475275724027,
                                      0.180546565876411, -0.013527027345810, 0.095775711512330, 0.211503307384337,
                                      0.203516673664514, -0.319739084548057, -0.404641931027738, 0.255067925735509,
                                      0.236808217763804, 0.269924877146794, -0.135338190717476, -0.037175478406564,
                                      -0.228640649357215, 0.396254760891929, 0.482420365803323, -0.455821220872313},
                                  {});

    clayer->prepare();
    clayer->feedforward();

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);
    clayer->backprop();

    bool chkKernelChanges = checkKernel({{1.524524050993934, 1.799336027071535, 0.282533307169485,
                                                 1.263240282796218, 1.447910803528541, 0.569016473650094,
                                                 1.834272867855236, 0.589841899502014, -0.560314856756130}
                                        }, clayer);
    if (!chkKernelChanges) {
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({ -6.240075187800725}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }
    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_1_ds_sd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8);

    clayer->init();
    clayer->prepare();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * 1, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * 1, 0.5);
    auto X = applyMask(Xflat, Xmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * 1);
    input->setOutput(0, X, Xmask);

    clayer->displayKernel();
    clayer->displayBias();
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {0, -0.013900942245441, -0.011262058319560, 0.020405191376393,
                                 0.017831183075847, 0.173864808523127, -0.055507110190687, -0.011512335946001,
                                 0.029611513653725, -0.352116463053354, -0.038547279395444, 0.116465123607043,
                                 -0.036364058124206, -0.032264677540081, 0.327628309612056, -0.330882176499300,
                                 -0.053018889624282, 0.078152735120199, 0.189887158897095, -0.133330767354904,
                                 -0.044148836678201, -0.009932782567500, 0.136359281535031, 0.368393325548477,
                                 -0.025545574072185, 0.186499859747414, 0.291057876039340, -0.110491886534209,
                                 0.031812249547410, 0.083008766926389, 0.015562944671884, -0.180775283868216}, {});

    auto dYflat = RandomDevice::createUniformVector(4 * 8 * 1, -1.0, 1.0);
    auto dYmask = RandomDevice::createMask(4 * 8 * 1, 0.5);
    auto dY = applyMask(dYflat, dYmask);
    //auto dYfull = makeDense(X,Xmask,4*8*1);

    output->setErrorSignal(0, dY, dYmask);
    input->setExpectedErrorSignal(0, {-0.022305119811452, 0.021844859601149, -0.113133430894785, 0.130367973980559,
                                      0.055947337265085, 0.061340929246879, 0.082178019603301, -0.061636803091091,
                                      0.183943782448853, 0.393603283216365, -0.148424595837797, -0.055919542618320,
                                      -0.049581453027529, -0.090837232767303, -0.311002369915301, -0.015185849188953,
                                      -0.209827767979636, -0.536298345751543, 0.229693214195886, 0.330185416496029,
                                      0.085183812395561, 0.245899662700232, -0.007159507769086, -0.135669287083880,
                                      -0.138138357436391, 0.277801854289113, -0.002430285317602, -0.052865979124757,
                                      -0.187878590996233, 0.357129012552227, -0.046653687738785, -0.184478186332896},
                                  {});

    clayer->prepare();
    clayer->feedforward();

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);
    clayer->backprop();

    bool chkKernelChanges = checkKernel({{-0.137343903015117, 0.473206292522991, 0.079150450383655,
                                                 0.158730469601765, 0.134585478264078, 0.410586791447990,
                                                 0.554936339492818, 0.176802516924769, 0.600258501550946,
                                         }
                                        }, clayer);
    if (!chkKernelChanges) {
        sDebug() << "Gradient calculation not passed (Kernel)";
    }

    bool chkBiasChanges = checkBias({  -1.375343021687832}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }
    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

//           --------------------------------------------------------- multiple out channel ----------------------------------------

bool __TestSConvolutionLayer::cycle_1_dd_dd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * 2);

    clayer->init();

    clayer->prepare();

    input->setOutput(0, RandomDevice::createUniformVector(4 * 8, -1.0, 1.0), {});

    input->displayOutput(0);
    clayer->displayBias();
    clayer->displayKernel();


    output->setExpectedInput(0, {       // Output cannel 1
            0.174691458836913, 0.435961600884787, -0.107943057251768, -0.259207117896071,
            -0.085117853690684, -0.271128986811996, -0.119489728179652, 0.127050927540238,
            0.016644381130684, 0.054198190289294, 0.491738580628493, 0.197566816235903,
            -0.130947968298303, -0.077715983348498, -0.019657028927782, -0.147295297651498,
            0.012592198080498, 0.039206565366330, -0.146431664177862, 0.248390739568529,
            0.107379351524102, -0.028370436381323, -0.116374968476822, -0.213297004468454,
            -0.183478584799954, -0.287192940150101, 0.017680026511883, -0.078441970615434,
            -0.005477596718356, 0.034128237292696, -0.039107803205355, 0.252010004266720,
            // output channel 2

            -0.027119544285150, -0.235679504177097, 0.171634884763705, 0.327092586598041,
            0.032194431941414, 0.062489590599189, 0.022955468805697, -0.042524046927126,
            -0.049807825079007, 0.014738244399056, -0.237992596496095, -0.097086811299819,
            0.030447798239560, 0.009549641982012, -0.144922522477187, -0.002832741986104,
            0.043246440504914, -0.129430534997236, 0.107892776737244, -0.326500132046806,
            -0.057760293571229, 0.029637979748559, -0.010816245832556, 0.086026255863516,
            0.006024425036745, 0.153633826685022, -0.271983866468872, 0.014324572155052,
            0.006729762563532, -0.077871691398030, 0.029429787655963, -0.236942206422550,

    }, {});

    output->setErrorSignal(0, RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0), {});

    input->setExpectedErrorSignal(0, {0.194144269776326, 0.069817736462865, -0.039787364244841, -0.031400982800522,
                                      0.136828188794151, -0.276350893838955, 0.128421143295668, 0.202022276382319,
                                      0.303287778305120, -0.249129777795360, -0.078128465213418, 0.178111573422775,
                                      -0.021133443603632, 0.092618555982819, 0.522106731183853, -0.156391741375908,
                                      -0.326171895287271, 0.255161958873241, 0.409971737898485, -0.018738790825130,
                                      -0.206221264044181, -0.133259053469482, -0.320583302146995, 0.085550633832620,
                                      -0.325875014603679, 0.356307630601810, -0.095601977621252, -0.019016665780362,
                                      -0.111296922967646, -0.160071580443870, -0.160934795506356, -0.004111182196169},
                                  {});

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    clayer->prepare();
    clayer->feedforward();

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{0.064665702571024,  0.623468829017785,  0.193126796578627,
                                                 -1.405081574151039, 0.569708142087776,  -1.413817549040721,
                                                 1.844461651726549, 0.626556058937633, -0.298032651128686},
                                         {-0.405918956128704, -0.679427897053677, 1.207429821999390,
                                                 -0.586980345742179, -1.165567415008707, -1.334192624432704,
                                                 1.595233122836301, 1.165435429241464, 0.289664023079322}
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({  -2.884311932191745,   -3.914688346242606}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_1_dd_sd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * 2);

    clayer->init();
    clayer->displayBias();
    clayer->displayKernel();
    clayer->prepare();

    input->setOutput(0, RandomDevice::createUniformVector(4 * 8, -1.0, 1.0), {});
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {       // Output cannel 1
            0.174691458836913, 0.435961600884787, -0.107943057251768, -0.259207117896071,
            -0.085117853690684, -0.271128986811996, -0.119489728179652, 0.127050927540238,
            0.016644381130684, 0.054198190289294, 0.491738580628493, 0.197566816235903,
            -0.130947968298303, -0.077715983348498, -0.019657028927782, -0.147295297651498,
            0.012592198080498, 0.039206565366330, -0.146431664177862, 0.248390739568529,
            0.107379351524102, -0.028370436381323, -0.116374968476822, -0.213297004468454,
            -0.183478584799954, -0.287192940150101, 0.017680026511883, -0.078441970615434,
            -0.005477596718356, 0.034128237292696, -0.039107803205355, 0.252010004266720,
            // output channel 2
            -0.027119544285150, -0.235679504177097, 0.171634884763705, 0.327092586598041,
            0.032194431941414, 0.062489590599189, 0.022955468805697, -0.042524046927126,
            -0.049807825079007, 0.014738244399056, -0.237992596496095, -0.097086811299819,
            0.030447798239560, 0.009549641982012, -0.144922522477187, -0.002832741986104,
            0.043246440504914, -0.129430534997236, 0.107892776737244, -0.326500132046806,
            -0.057760293571229, 0.029637979748559, -0.010816245832556, 0.086026255863516,
            0.006024425036745, 0.153633826685022, -0.271983866468872, 0.014324572155052,
            0.006729762563532, -0.077871691398030, 0.029429787655963, -0.236942206422550,
    }, {});

    auto dYproto = RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0);
    auto dYmask = RandomDevice::createMask(4 * 8 * 2, 0.5);
    auto dYpartial = applyMask(dYproto, dYmask);

    output->setErrorSignal(0, dYpartial, dYmask);
    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {-0.244206335574466, 0.306145988141458, 0.051233518183238, -0.045134607599280,
                                      -0.096174289196094, 0.087521683830480, -0.290200063467504, -0.014320045313033,
                                      0.043782804077714, 0.236667396676361, 0.129764807857002, 0.122691518518947,
                                      0.116263552933398, 0.037033078983652, -0.137150017005511, 0.017396606809344,
                                      0.119967451677707, -0.138783085428872, 0.235244141927567, -0.023819149276755,
                                      -0.159534792986719, -0.104795539444862, 0.122363694810449, -0.086137875528566,
                                      -0.032249655173166, -0.057790223655304, 0.069925673541994, 0.127891561720704,
                                      -0.112901196891924, 0.329134611895457, 0.081801360102516, 0.019621606048722}, {});

    clayer->prepare();
    clayer->feedforward();

    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-1.471983939694808, -1.561028559352307, -0.304013845772236,
                                                 1.608222002227861,  0.355476765881821,  0.143359603815767,
                                                 -0.558186513309113, -1.465728227997082, -1.136749986523331},
                                         {0.250799884178733,  -1.808688927541378, -1.439555140797433,
                                                 -1.198268346285196, -0.628181471884770, 0.710405025640647,
                                                 3.504884029782295,  -0.972450469994395, -1.310842653832281}
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({    -2.269564379822341,    -0.987421635070023}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_1_ds_dd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * descConv.inChannel, 0.5);
    //auto dYmask = RandomDevice::createMask(4*8*descConv.outChannel,0.5);
    auto X = applyMask(Xflat, Xmask);
    //auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * descConv.inChannel);
    //auto dYfull = makeDense(dY, dYmask, 4*8*descConv.outChannel);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.088420392534866, -0.089196060538796, 0.227219452592575, 0.145607642197070, 0.193836281042852,
            0.281544937309472, -0.089558186708959, -0.069904079794495, -0.108022312096553, -0.243687435954734,
            -0.154310828909034, 0.076623106829522, -0.040729023178368, -0.071755383708160, 0.225036558770463,
            0.245963035588719, 0.001303939224883, -0.000739019538222, 0.027101930120240, 0.024827086106601,
            -0.002714994695738, 0.000468958174179, 0.001216041095287, 0, -0.045298495987085, -0.006289591822353,
            -0.070481395257852, 0.129908584779662, 0.105904184927111, 0.047562700961977, 0.170219356332717,
            -0.173806393055055,

            // output channel 2
            0.023158008559094, 0.095099097599783, -0.131551172076239, -0.041838802216256, -0.048858537238521,
            -0.158483452034869, 0.157087574156563, 0.079967766952202, 0.034346981018647, 0.038699246748945,
            -0.013934536485612, -0.088342878395879, -0.003820408340011, 0.039667998687111, -0.144823145821596,
            -0.173245452143261, -0.001232764535329, 0.000380596156194, 0.003476482196895, -0.029581286256198,
            0.000729363892185, -0.001727923891960, -0.001529191123969, 0, 0.069635892369944, -0.052515685185180,
            0.110014798608861, -0.121352730791137, -0.031962151657480, 0.101859980561067, -0.051372633535581,
            0.218564318738063
    }, {});

    output->setErrorSignal(0, dYflat, {});

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
                                      0.194144269776326, 0.069817736462865,-0.039787364244841,-0.031400982800522
                                      , 0.136828188794151,-0.276350893838955, 0.128421143295668, 0.202022276382319
                                      , 0.303287778305120,-0.249129777795360,-0.078128465213418, 0.178111573422775
                                      ,-0.021133443603632, 0.092618555982819, 0.522106731183853,-0.156391741375908
                                      ,-0.326171895287271, 0.255161958873241, 0.409971737898485,-0.018738790825130
                                      ,-0.206221264044181,-0.133259053469482,-0.320583302146995, 0.085550633832620
                                      ,-0.325875014603679, 0.356307630601810,-0.095601977621252,-0.019016665780362
                                      ,-0.111296922967646,-0.160071580443870,-0.160934795506356,-0.004111182196169
            }, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             1.090586575560487,  -0.722731696624436,  -0.328862763248930,
                                             1.782582684493365,  -1.797513134483713,   0.531641392805135,
                                             1.536521385577093,  -0.280885003355565,   0.159800895561526

                                         },
                                         {         -1.441029639062602,  -0.819168286350038,  -0.756914217894980,
                                                   -0.212349673360811,  -1.167041683877690,   0.270094798811246,
                                                    0.725642747241685,  -0.019754203928173,  -0.772605128771478
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-2.884311932191745,  -3.914688346242606}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_1_ds_sd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 1;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * descConv.inChannel, 0.5);
    auto dYmask = RandomDevice::createMask(4 * 8 * descConv.outChannel, 0.5);
    auto X = applyMask(Xflat, Xmask);
    auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * descConv.inChannel);
    auto dYfull = makeDense(dY, dYmask, 4 * 8 * descConv.outChannel);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.088420392534866, -0.089196060538796, 0.227219452592575, 0.145607642197070, 0.193836281042852,
            0.281544937309472, -0.089558186708959, -0.069904079794495, -0.108022312096553, -0.243687435954734,
            -0.154310828909034, 0.076623106829522, -0.040729023178368, -0.071755383708160, 0.225036558770463,
            0.245963035588719, 0.001303939224883, -0.000739019538222, 0.027101930120240, 0.024827086106601,
            -0.002714994695738, 0.000468958174179, 0.001216041095287, 0, -0.045298495987085, -0.006289591822353,
            -0.070481395257852, 0.129908584779662, 0.105904184927111, 0.047562700961977, 0.170219356332717,
            -0.173806393055055,

            // output channel 2
            0.023158008559094, 0.095099097599783, -0.131551172076239, -0.041838802216256, -0.048858537238521,
            -0.158483452034869, 0.157087574156563, 0.079967766952202, 0.034346981018647, 0.038699246748945,
            -0.013934536485612, -0.088342878395879, -0.003820408340011, 0.039667998687111, -0.144823145821596,
            -0.173245452143261, -0.001232764535329, 0.000380596156194, 0.003476482196895, -0.029581286256198,
            0.000729363892185, -0.001727923891960, -0.001529191123969, 0, 0.069635892369944, -0.052515685185180,
            0.110014798608861, -0.121352730791137, -0.031962151657480, 0.101859980561067, -0.051372633535581,
            0.218564318738063
    }, {});

    output->setErrorSignal(0, dY, dYmask);

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
                                      0.284595701398666,   0.078330184276215,   0.217679421026877,  -0.059830521439204,
                                      0.092580959492298,  -0.144293548902413,   0.074670568111738,  -0.032338270992256,
                                     -0.046469021015330,  -0.031281084353699,   0.002309915141586,   0.157068082054637,
                                      0.182232853091996,  -0.137129114959637,  -0.005952680969758,   0.092215382002398,
                                      0.065761907223407,  -0.009651158981934,  -0.151047073485425,   0.004433206416332,
                                     -0.114121221741993,   0.347672751799904,  -0.067688683635590,  -0.216454970559215,
                                      0.153239554394060,   0.371877686133627,  -0.315058410410585,   0.029680822610790,
                                     -0.016205855679594,   0.552751373765614,   0.252950970296277,  -0.189895635423796}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             0.167443349136707,  -0.133853541230070,   0.292515578843206,
                                             1.282831983916477,  -0.604771556599089,   0.297032266302742,
                                             0.045756671772679,  -0.689607615158925,  -0.304885073231934
                                         },
                                         {        -0.679200576476434,  -0.139322354774200,  -0.552672193129063,
                                                  0.374514972217413,   1.497434508607669,   0.441165034450574,
                                                  0.182774234791344,  -0.296890646853787,  -0.067323060517457
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({ -2.462196162270384,-2.409789657419789}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}
// -------------------------------------------------------------------- multiple in channel ----------------------------------------
//           ---------------------------------------------------------  multiple out channel ---------------------------------------

bool __TestSConvolutionLayer::cycle_2_dd_dd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * 2);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * 2);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    input->setOutput(0, RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0), {});
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.083590011955671, 0.523087913299441, -0.103067270885265, -0.029613138204533,
            -0.093074914091776, -0.067222493232364, -0.322312243802413, 0.085258956026425,
            -0.286086383310417, -0.039769830510539, 0.268368846588107, 0.090265072614775,
            0.043967966897931, 0.020940922499857, 0.413535963079910, -0.226598427501299,
            -0.096569974100748, 0.141514094863524, 0.226754792565783, 0.068800492339689,
            0.035602786600721, -0.016657052975164, 0.013503224144623, 0.353498316042249,
            0.317998614733131, -0.097541104273438, -0.716812796532532, 0.207331280428936,
            0.047854295053041, -0.176554925509343, 0.053202965074773, 0.111510889703280,

            // output channel 2
            0.001364218365751, -0.125071750477941, 0.043093511110380, -0.072560484739950,
            -0.066116373635705, -0.394878870929935, 0.067878766571264, -0.213941236097793,
            -0.038864608460387, -0.109469934460309, 0.439669881413865, 0.269239011240997,
            0.079578865145544, 0.216546164514918, -0.521446525065448, 0.193045550146722,
            -0.100487779388125, 0.018677493052309, 0.499182871910124, -0.071761109632426,
            0.421341968933943, -0.324435970741011, -0.762576530721552, -0.011484879798573,
            0.191640292746598, 0.205470799633275, -0.570098932137979, 0.007592015503468,
            0.018955409103401, -0.015709375266698, -0.101259345829992, 0.187858129485173
    }, {});

    output->setErrorSignal(0, RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0), {});
    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {0.100788761062986, -0.219794296215658, 0.371206266165810, -0.048317490244575,
                                      -0.141507438698241, -0.243738746527607, 0.610443929793561, -0.140491776628809,
                                      0.070295774030575, -0.172678314678981, -0.177286820718385, -0.210154997674851,
                                      -0.495729473850961, 0.048513340374082, -0.183116502008122, -0.184814803788747,
                                      -0.039800171518589, 0.109127180874352, 0.188043219574662, 0.121362355234095,
                                      -0.019536078712392, -0.008481779278925, 0.565621897698876, -0.196778198298607,
                                      0.594610172507529, -0.053123895641386, 0.138271921796473, -0.377800882599647,
                                      -0.222480360620958, -0.455546113627048, -0.141496302359319, -0.111189555623215,

                                      0.085488521160360, 0.148316995183120, -0.329992994110633, 0.041621853866203,
                                      0.010197802176838, 0.430578835239824, 0.392478133182436, 0.111384910931755,
                                      -0.156305670246681, -0.464820619160200, -0.015633569266199, -0.206995752771076,
                                      -0.057276550783323, -0.065196399623343, 0.197331436918502, -0.456234843982218,
                                      0.575556018755474, -0.090227622542795, 0.420517344718947, -0.043919354907470,
                                      -0.437056114388599, -0.140070959221726, 0.045832458753886, 0.116127089925593,
                                      -0.215120165635575, 0.082631000227062, 0.026845425833858, 0.004535741212849,
                                      0.151748769419943, -0.256349719544404, -0.411587328250138, -0.229427688514925},
                                  {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{-2.062368261616364, 2.319320976404915,  -0.392115219122477,
                                                 1.207047319428557, 0.498290628878299, 0.207928878683585,
                                                 0.990127930181164, 1.530534559430986, 0.261699293254602,
                                                 1.391364117716604, 0.248737851880550, 1.233436797699110,
                                                 -0.779314555750494, 1.486904067578425,  2.265294849845327,
                                                 -0.071259733001794, -1.565588473643008, -1.441230732484847
                                         },
                                         {-1.411193202325384, -0.946273585020807, 0.597691185824319,
                                                 2.925320929790312, 2.656370836367171, 1.148414178152073,
                                                 0.645035766380910, 1.755169941661191, -0.732632171049806,
                                                 1.145488580465582, 1.311382220295297, 1.950559823880961,
                                                 1.661407703168605,  -1.340368096002400, -1.973110031233646,
                                                 0.051030078391260,  0.121929258292024,  -1.942260587437324}
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({0.783631690040232, 1.227232347333451}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_2_dd_sd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    //auto Xmask = RandomDevice::createMask(4*8*2, 0.5);
    //auto X = applyMask(Xflat, Xmask);
    //auto Xfull = makeDense(X,Xmask,4*8*2);

    input->setOutput(0, Xflat, {});
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.083590011955671, 0.523087913299441, -0.103067270885265, -0.029613138204533,
            -0.093074914091776, -0.067222493232364, -0.322312243802413, 0.085258956026425,
            -0.286086383310417, -0.039769830510539, 0.268368846588107, 0.090265072614775,
            0.043967966897931, 0.020940922499857, 0.413535963079910, -0.226598427501299,
            -0.096569974100748, 0.141514094863524, 0.226754792565783, 0.068800492339689,
            0.035602786600721, -0.016657052975164, 0.013503224144623, 0.353498316042249,
            0.317998614733131, -0.097541104273438, -0.716812796532532, 0.207331280428936,
            0.047854295053041, -0.176554925509343, 0.053202965074773, 0.111510889703280,

            // output channel 2
            0.001364218365751, -0.125071750477941, 0.043093511110380, -0.072560484739950,
            -0.066116373635705, -0.394878870929935, 0.067878766571264, -0.213941236097793,
            -0.038864608460387, -0.109469934460309, 0.439669881413865, 0.269239011240997,
            0.079578865145544, 0.216546164514918, -0.521446525065448, 0.193045550146722,
            -0.100487779388125, 0.018677493052309, 0.499182871910124, -0.071761109632426,
            0.421341968933943, -0.324435970741011, -0.762576530721552, -0.011484879798573,
            0.191640292746598, 0.205470799633275, -0.570098932137979, 0.007592015503468,
            0.018955409103401, -0.015709375266698, -0.101259345829992, 0.187858129485173
    }, {});

    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto dYmask = RandomDevice::createMask(4 * 8 * descConv.outChannel, 0.5);
    auto dY = applyMask(dYflat, dYmask);
    //auto dYfull = makeDense(dY, dYmask, 4*8*2);

    output->setErrorSignal(0, dY, dYmask);

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);
    input->setExpectedErrorSignal(0, {0.266452104247821, 0.351278209584464, -0.503252410225138, 0.248136195380520,
                                      0.060029628623840, -0.313904947488050, -0.083762351207850, 0.070908769333885,
                                      0.065980434458656, -0.025604317048699, 0.076279500330426, -0.047224916731011,
                                      0.101763190287857, -0.148627984918717, -0.131009964739991, -0.088259104150623,
                                      -0.344794041089001, -0.086041205169719, 0.171990845841758, -0.024614180671204,
                                      0.170846441192442, 0.108620660893854, -0.065842636181898, -0.123217551523082,
                                      -0.263931145989278, -0.430613775372516, 0.081009977007946, 0.067739125233263,
                                      0.231804935894038, 0.447109136898143, 0.058658035456261, -0.114232283709002,

                                      -0.149584813141811, -0.134582480104926, 0.010064399073946, 0.079487034688749,
                                      -0.104470212189692, 0.107931084479533, 0.289255207570855, 0.151348647874480,
                                      0.206348584588891, -0.034602962633485, -0.023985224874845, 0.045786644719878,
                                      -0.087892588053925, -0.060418275176243, 0.129979573202242, 0.016244254336206,
                                      -0.174645737182570, -0.151349874325899, -0.097238016753262, 0.095249127125039,
                                      -0.035679376572601, 0.248806097954979, -0.174328862078120, -0.089867823057957,
                                      -0.014702480160685, 0.048226947987489, -0.285538298198506, -0.105975454903475,
                                      0.080665345098045, 0.044615952486229, 0.227230556932543, 0.055457639934486}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                                 0.251025838430097,  0.967633321514845, 0.566407761505108,
                                                 0.689353680475637,  -0.430772834198879, 0.659202877525651,
                                                 -0.170612334530721, 0.160547887642081, 1.567850900554411,

                                                 -0.733926258344699, -0.062700573858369, 0.253051992411711,
                                                 0.230613247038577,  0.126028011916269,  0.736095734168713,
                                                 0.115527615503072, 1.702042281385570,  -0.386574598738029
                                         },
                                         {       -1.189069667128219, 0.069342898903703, 0.404687522576125,
                                                 -0.993107229185342, -1.093553544380643, 2.644108481613122,
                                                 0.387574138170108,  0.701357797666493, 1.235031797041764,

                                                 -2.157186131495219, -0.217214909149873, 0.129868964773042,
                                                 -1.054176862854603, -1.383028381784351, 0.130918272710431,
                                                 1.329609478633945, -1.302141094671490, 1.832644217407971
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-0.545535069597496, 1.270202457334892}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_2_ds_dd_2() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * descConv.inChannel, 0.5);
    //auto dYmask = RandomDevice::createMask(4*8*descConv.outChannel,0.5);
    auto X = applyMask(Xflat, Xmask);
    //auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * descConv.inChannel);
    //auto dYfull = makeDense(dY, dYmask, 4*8*descConv.outChannel);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.123288357609995, 0.301980589605635, 0.257515408865661, -0.252964360036999,
            -0.072648975440323, -0.019069531960598, 0.122421059316258, 0.049027841006028,
            -0.137100584794402, 0.122141231947358, -0.444182061307059, 0.269450105356494,
            -0.031420716073184, 0.169521363661365, -0.123521024551195, 0.268446476832641,
            0.173114571413652, -0.302006955219978, 0.186665872078588, -0.169137324260009,
            -0.103792846627348, -0.044812199440326, -0.090096811081646, 0.316074722196333,
            -0.077217422756947, 0.290034250503272, 0.134147590325396, -0.009158826330498,
            0.141887042079763, -0.153661986278561, 0.105126874188901, -0.157266774512773,

            // output channel 2
            -0.145775079437104, 0.162836948569692, 0.218826702851702, 0.033497236387529,
            -0.204942167529217, -0.074337451427945, -0.587823640732119, -0.081319403199409,
            -0.297426601222791, 0.356020099775463, 0.116144197852660, -0.025520459298736,
            0.177805758060709, 0.292819839881851, -0.282443465117338, 0.326870545062853,
            -0.344700423081608, -0.086010480245144, 0.131255945511349, 0.029630578603508,
            0.137284937431877, 0.109765211510298, -0.150359939544978, -0.047897007310655,
            0.027328632785853, -0.041415050964087, 0.138423933389230, 0.002306297778949,
            -0.021093899041268, -0.172831055467036, 0.105200083573140, -0.014331608083826
    }, {});

    output->setErrorSignal(0, dYflat, {});

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
                                      0.100788761062986,  -0.219794296215658,   0.371206266165810,  -0.048317490244575,
                                     -0.141507438698241,  -0.243738746527607,   0.610443929793561,  -0.140491776628809,
                                      0.070295774030575,  -0.172678314678981,  -0.177286820718385,  -0.210154997674851,
                                     -0.495729473850961,   0.048513340374082,  -0.183116502008122,  -0.184814803788747,
                                     -0.039800171518589,   0.109127180874352,   0.188043219574662,   0.121362355234095,
                                     -0.019536078712392,  -0.008481779278925,   0.565621897698876,  -0.196778198298607,
                                      0.594610172507529,  -0.053123895641386,   0.138271921796473,  -0.377800882599647,
                                     -0.222480360620958,  -0.455546113627048,  -0.141496302359319,  -0.111189555623215,

                                      0.085488521160360,   0.148316995183120,  -0.329992994110633,   0.041621853866203,
                                      0.010197802176838,   0.430578835239824,   0.392478133182436,   0.111384910931755,
                                     -0.156305670246681,  -0.464820619160200,  -0.015633569266199,  -0.206995752771076,
                                     -0.057276550783323,  -0.065196399623343,   0.197331436918502,  -0.456234843982218,
                                      0.575556018755474,  -0.090227622542795,   0.420517344718947,  -0.043919354907470,
                                     -0.437056114388599,  -0.140070959221726,   0.045832458753886,   0.116127089925593,
                                     -0.215120165635575,   0.082631000227062,   0.026845425833858,   0.004535741212849,
                                      0.151748769419943,  -0.256349719544404,  -0.411587328250138,  -0.229427688514925}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             0.229505498165251,   0.122822165772520,   0.649001997606393,
                                            -0.787562061513983,  -1.185561607136890,   0.413781743166068,
                                            -0.124292863673085,   1.208951428291607,   0.552400025326503,

                                             2.121111698552055,   0.422971821994885,   0.403300605441395,
                                            -0.092416292720674,   1.090396814222751,   0.716014798335508,
                                            -0.583670447553281,  -0.301044544106426,  -1.148524775626626,

                                         },
                                         {
                                             0.955628574617395,   1.836149277206310,   1.000297194738950,
                                             2.019596568978471,   1.054961073852464,   0.191648879871369,
                                             0.588237005965676,  -2.350884418708550,   1.845660240378979,

                                            -1.183037265838725,   2.105121940859204,   1.589502342818409,
                                             1.242925379651391,  -1.356080046015684,  -1.589453755068786,
                                            -0.952729330552573,  -0.906477252780380,   0.275941481273851

                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({0.783631690040232, 1.227232347333451}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

bool __TestSConvolutionLayer::cycle_2_ds_sd_2() { ////// todoooooo o ! oo ! ooo ! ooo ! ooo !
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 2;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * 2);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * 2);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * 2, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * 2, 0.5);
    auto dYmask = RandomDevice::createMask(4 * 8 * 2, 0.5);
    auto X = applyMask(Xflat, Xmask);
    auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * 2);
    auto dYfull = makeDense(dY, dYmask, 4 * 8 * 2);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.123288357609995, 0.301980589605635, 0.257515408865661, -0.252964360036999,
            -0.072648975440323, -0.019069531960598, 0.122421059316258, 0.049027841006028,
            -0.137100584794402, 0.122141231947358, -0.444182061307059, 0.269450105356494,
            -0.031420716073184, 0.169521363661365, -0.123521024551195, 0.268446476832641,
            0.173114571413652, -0.302006955219978, 0.186665872078588, -0.169137324260009,
            -0.103792846627348, -0.044812199440326, -0.090096811081646, 0.316074722196333,
            -0.077217422756947, 0.290034250503272, 0.134147590325396, -0.009158826330498,
            0.141887042079763, -0.153661986278561, 0.105126874188901, -0.157266774512773,

            // output channel 2
            -0.145775079437104, 0.162836948569692, 0.218826702851702, 0.033497236387529,
            -0.204942167529217, -0.074337451427945, -0.587823640732119, -0.081319403199409,
            -0.297426601222791, 0.356020099775463, 0.116144197852660, -0.025520459298736,
            0.177805758060709, 0.292819839881851, -0.282443465117338, 0.326870545062853,
            -0.344700423081608, -0.086010480245144, 0.131255945511349, 0.029630578603508,
            0.137284937431877, 0.109765211510298, -0.150359939544978, -0.047897007310655,
            0.027328632785853, -0.041415050964087, 0.138423933389230, 0.002306297778949,
            -0.021093899041268, -0.172831055467036, 0.105200083573140, -0.014331608083826
    }, {});

    output->setErrorSignal(0, dY, dYmask);

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
            0.181838647514085, -0.112519365321052, 0.408281886036690, -0.079657194607206,
            -0.032552994066543, -0.193476681379172, -0.018775384714662, 0.071223195203877,
            0.091631056062678, -0.153275887908643, -0.143382052661884, 0.135173696535431,
            0.167902931555063, -0.076590312465672, 0.241904676378604, 0.234298977800688,
            -0.038547533416715, 0.095105954600457, 0.075794957858222, -0.227293065196684,
            -0.032593910918599, -0.001370704203821, 0.191043968423765, 0.230456568109844,
            -0.223537763611338, 0.302523388896628, -0.164258855737689, -0.189430492746588,
            -0.179941877009355, -0.036675798959553, -0.025314673053069, -0.002779594862406,

            0.004165907447363, -0.057568308136688, -0.011568583745170, -0.178312976572353,
            0.219299090762309, -0.203320777568612, -0.081115642445966, -0.093213734090326,
            -0.164837940469914, 0.351039246970672, 0.280369561801670, -0.198687047343080,
            -0.226839817529057, -0.094014522246996, 0.106408656742734, 0.185090714235080,
            -0.035018073466595, 0.035174589558418, -0.285517163830486, -0.008177531796749,
            -0.062630827878539, 0.146345514983865, 0.278087841813650, 0.213343222521731,
            0.148937410606916, 0.180544428225898, -0.104385226071343, -0.107439896190432,
            0.041339611322326, -0.093422841483310, -0.045589457007238, 0.020658550975737}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                                 -0.760846979058988, -0.047054162770988, -0.480998000693492,
                                                 -0.085468769610104, -0.089863360452634, 0.552808968678509,
                                                 0.624820681327792, -0.317427726907304, -0.321025632255318,

                                                 0.210804223397307,  -0.102625025831848, -0.175870843293917,
                                                 0.561324366887418,  0.055772414052659, -0.315442624975032,
                                                 -0.886667042482887, 0.450062398224274,  1.003076475496888

                                         },
                                         {       0.169589832543290,  -1.753014586790842, -0.951540890208407,
                                                 0.367450322777613,  -2.702845740479435, -0.595786941798273,
                                                 0.133868277315966, -0.318471923814567, -1.282559612699280,

                                                 -0.089056461234873, -0.593449767478568, -0.162155074726420,
                                                 -0.218363349154249, 1.610635002717041, 0.303074749142472,
                                                 0.457079670646958,  -1.810157143032379, -0.542067215840190,
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({1.203566198454039,-0.818559007012642}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

//           ---------------------------------------------------------            1 channel ----------------------------------------

// miss
bool __TestSConvolutionLayer::cycle_2_dd_dd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    //auto Xmask = RandomDevice::createMask(4*8*descConv.inChannel, 0.5);
    //auto dYmask = RandomDevice::createMask(4*8*descConv.outChannel,0.5);
    //auto X = applyMask(Xflat, Xmask);
    //auto dY = applyMask(dYflat, dYmask);
    //auto Xfull = makeDense(X,Xmask,4*8*descConv.inChannel);
    //auto dYfull = makeDense(dY, dYmask, 4*8*descConv.outChannel);

    input->setOutput(0, Xflat, {});
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            0.240803475596060, 0.934707141905398, -0.150655119514029, -0.288766414574110, -0.324564109224385,
            -0.092864659901618, -0.025918713017169, 0.472928657852860, -0.049539160289959, -0.080886654001786,
            0.818061006169738, 0.421578690321460, -0.097337787755106, -0.265156102415395, -0.291597770655013,
            -0.101501900940817, 0.085650127023655, -0.472137310399195, -0.018536141832128, 0.926389273746348,
            0.290858350482473, -0.046224929902362, -0.238883070790080, -0.025841862738535, -0.471869624948045,
            -0.012939096477664, -0.006587500707412, 0.042459379892842, 0.002821834890277, -0.096554206801746,
            0.244824108506589, 0.021306839316731
    }, {});

    output->setErrorSignal(0, dYflat, {});

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {-0.096054520134699,  -0.029288987357290,   0.340703975038030,   0.003497465613666,
                                      -0.206216523226878,   0.428851691952449,   0.065782164204160,   0.180745885500210,
                                      -0.208679539960644,   0.095775711512329,   0.350659447623224,  -0.112761015898973,
                                      -0.164160620144307,  -0.404641931027738,  -0.008485667011137,   0.331071844264977,
                                       0.111618314872941,  -0.135338190717476,  -0.014358296105890,   0.101518719522356,
                                       0.546164431501642,   0.472840177430353,  -0.349745779175225,  -0.420886688575794,
                                       0.245279243215589,  -0.184800438565583,  -0.080214492909388,   0.106916775047764,
                                      -0.082904082709605,  -0.070347952533887,  -0.186914029644301,  -0.031752398162927,

                                      0.163914869583781,   0.215676447545632,  -0.195479415730225,  -0.007964988640376,
                                      0.419224830602448,  -0.105513970405543,   0.239806635630082,  -0.128702798321964,
                                      0.450435222973330,   0.163351814955594,  -0.184402849871988,   0.105765593221386,
                                      0.270852210361867,   0.512187592606847,   0.192533644939793,  -0.146269124193396,
                                     -0.257876316824527,   0.140992722424362,   0.129622232574581,   0.012713307680350,
                                     -0.555822620578800,  -0.412932392760740,   0.081496159323806,   0.159116435675688,
                                     -0.182945376556880,   0.062592527674468,  -0.073218246826691,  -0.073535330154151,
                                      0.038276704809107,  -0.155712308580444,   0.102884136720407,  -0.0104869558219347}, {});
    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{     -0.405918956128704,  -0.679427897053677,   1.207429821999391,
                                               -0.586980345742179,  -1.165567415008707,  -1.334192624432704,
                                                1.595233122836301,   1.165435429241463,   0.289664023079322
                                         },
                                         {         2.900746953881693,  -0.516714651591065,  -1.761050804984475,
                                                   0.376747855827462,  2.759987544337184,   0.057700361042216,
                                                   2.154075711753711,   0.792863214430440  -1.606353826642195
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-3.914688346242606}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_2_dd_sd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    //auto Xmask = RandomDevice::createMask(4*8*descConv.inChannel, 0.5);
    auto dYmask = RandomDevice::createMask(4 * 8 * descConv.outChannel, 0.5);
    //auto X = applyMask(Xflat, Xmask);
    auto dY = applyMask(dYflat, dYmask);
    //auto Xfull = makeDense(X,Xmask,4*8*descConv.inChannel);
    auto dYfull = makeDense(dY, dYmask, 4 * 8 * descConv.outChannel);

    input->setOutput(0, Xflat, {});
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
                                   -0.268318940051302,   0.338659012665896,  -0.161921867972628,   0.029724514189631,
                                   -0.302631353461289,   0.306230528957212,  -0.207947041391768,   0.024128629652832,
                                   -0.070924133612356,  -0.104627550740200,   0.270393990884921,   0.117282501651647,
                                    0.149692724801392, -0.026219397813568,  -0.268381571046961,   0.094063102901481,
                                   -0.197658183342610,   0.099093432868952,   0.162022985573124,  -0.055804940680892,
                                   -0.065205301085579,   0.389774326287251,   0.212557402867822,                   0,
                                    0.114806101449895,  -0.055631170471391,  -0.148196535302938,  -0.030888677712184,
                                    0.042767358290103,   0.159441862201265,  -0.117840046804464,  -0.107787415213057,

            // output channel 2
                                   0.295140588311422,  -0.284046105413321,   0.298028979729891,  -0.072063430411847,
                                   0.354602589824629,  -0.227006844283760,   0.261489993490434,   0.003656448521186,
                                   0.131018144154378,   0.290247077301564,  -0.177349895153215,   0.053034352648479,
                                  -0.139833364790408,   0.075506802878504,   0.246458221230023,  -0.029416739559154,
                                   0.193235619075596,  -0.060096869855012,  -0.088737158743784,   0.086587157596506,
                                   0.246448963403528,  -0.104627184459467,  -0.046583760691655,                   0,
                                  -0.065047677438606,   0.055389181339566,   0.065850249074910,  -0.002897378285187,
                                  -0.012907279525864,  -0.159881482846647,   0.035564376296726,   0.023622527788315
    }, {});

    output->setErrorSignal(0, dY, dYmask);

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {  -0.268318940051302,   0.338659012665896,  -0.161921867972628,   0.029724514189631,
                                        -0.302631353461289,   0.306230528957212,  -0.207947041391768,   0.024128629652832,
                                        -0.070924133612356,  -0.104627550740200,   0.270393990884921,   0.117282501651647,
                                         0.149692724801392,  -0.026219397813568,  -0.268381571046961,   0.094063102901481,
                                        -0.197658183342610,   0.099093432868952,   0.162022985573124,  -0.055804940680892,
                                        -0.065205301085579,   0.389774326287251,   0.212557402867822,                   0,
                                         0.114806101449895,  -0.055631170471391,  -0.148196535302938,  -0.030888677712184,
                                         0.042767358290103,   0.159441862201265,  -0.117840046804464,  -0.107787415213057,

                                        0.295140588311422,  -0.284046105413321,   0.298028979729891,  -0.072063430411847,
                                        0.354602589824629,  -0.227006844283760,   0.261489993490434,   0.003656448521186,
                                        0.131018144154378,   0.290247077301564,  -0.177349895153215,   0.053034352648479,
                                       -0.139833364790408,   0.075506802878504,   0.246458221230023,  -0.029416739559154,
                                        0.193235619075596,  -0.060096869855012,  -0.088737158743784,   0.086587157596506,
                                        0.246448963403528,  -0.104627184459467,  -0.046583760691655,                   0,
                                       -0.065047677438606,   0.055389181339566,   0.065850249074910,  -0.002897378285187,
                                       -0.012907279525864,  -0.159881482846647,   0.035564376296726,   0.023622527788315,
                                  }, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             -2.608326938262357,  -1.346177477096151,  -0.434159398572767,
                                             -0.062281863441052,   0.673345001966615,   0.065542470106005,
                                             -0.247918406350349,   0.189473254577216,   0.315420953142884

                                         },
                                         {     1.079001785043540,   0.143949445373934,   2.919756887210088,
                                               0.623165924859395,  -0.121437192514473,   0.719674974109372,
                                               0.564214140524987,   0.522745621098592,   0.745541840921321

                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-4.133739505974657}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_2_ds_dd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * descConv.inChannel, 0.5);
    //auto dYmask = RandomDevice::createMask(4*8*descConv.outChannel,0.5);
    auto X = applyMask(Xflat, Xmask);
    //auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * descConv.inChannel);
    //auto dYfull = makeDense(dY, dYmask, 4*8*descConv.outChannel);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.194628311933751, -0.063598531542797, 0.170390511862326, 0.071969653895974, 0.333795059998075,
            0.160100539354932, -0.138584947998306, -0.086282996297902, -0.103990417583908, -0.560367345932559,
            -0.325404535703320, 0.342507206484013, -0.069879848178469, 0.004532028280636, 0.207631885034227,
            0.368325763195916, -0.039777696081293, -0.243648775387515, 0.323398380139467, -0.284983436803639,
            -0.082808785843329, 0.086628104435898, -0.079531911004458, -0.131838922109608, 0.062754258471725,
            -0.298706372285531, -0.284244645446337, 0.156836375254404, 0.190970992714785, 0.039503696245317,
            -0.024757100865504, -0.259552515827781

    }, {});

    output->setErrorSignal(0, dYflat, {});

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
                                      -0.096054520134699,  -0.029288987357290,   0.340703975038030,   0.003497465613666,
                                      -0.206216523226878,   0.428851691952449,   0.065782164204160,   0.180745885500210,
                                      -0.208679539960644,   0.095775711512329,   0.350659447623224,  -0.112761015898973,
                                      -0.164160620144307,  -0.404641931027738,  -0.008485667011137,   0.331071844264977,
                                       0.111618314872941,  -0.135338190717476,  -0.014358296105890,   0.101518719522356,
                                       0.546164431501642,   0.472840177430353,  -0.349745779175225,  -0.420886688575794,
                                       0.245279243215589,  -0.184800438565583,  -0.080214492909388,  0.106916775047764,
                                      -0.082904082709605,  -0.070347952533887,  -0.186914029644301, -0.031752398162927,

                                      0.163914869583781,   0.215676447545632,  -0.195479415730225,  -0.007964988640376,
                                        0.419224830602448,  -0.105513970405543,   0.239806635630082,  -0.128702798321964,
                                        0.450435222973330,   0.163351814955594,  -0.184402849871988,  0.105765593221386,
                                        0.270852210361867,   0.512187592606847,   0.192533644939793,  -0.146269124193396,
                                       -0.257876316824527,   0.140992722424362,   0.129622232574581,   0.012713307680350,
                                       -0.555822620578800,  -0.412932392760740,   0.081496159323806,   0.159116435675688,
                                       -0.182945376556880,   0.062592527674468,  -0.073218246826691,  -0.073535330154151,
                                        0.038276704809107,  -0.155712308580444,   0.102884136720407,  -0.010486955821934}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             -1.441029639062602,  -0.819168286350038,  -0.756914217894980,
                                             -0.212349673360811,  -1.167041683877690,   0.270094798811246,
                                              0.725642747241685, -0.019754203928173,  -0.772605128771478


                                         },
                                         {         -1.148788484328920,   0.440001417560365,  -0.190859848048917,
                                                   -0.732069528502938,  -0.552395607725970,   0.535629444588603,
                                                   -0.170220729589539,   0.178120955602511,  -0.751867315923102
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-3.914688346242606}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

// miss
bool __TestSConvolutionLayer::cycle_2_ds_sd_1() {
    RandomDevice::setSeed(1024);

    ConvolutionLayerDescription descConv;
    descConv.dimKernel = {3, 3};
    descConv.scatter = {1, 1};
    descConv.offset = {(descConv.dimKernel[0] - 1) / 2, (descConv.dimKernel[1] - 1) / 2};
    descConv.leftPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.rightPadding = {PaddingType::Zerofill, PaddingType::Zerofill};
    descConv.inChannel = 2;
    descConv.outChannel = 1;
    descConv.dimInput = {4, 8};
    descConv.lower = {0, 0};
    descConv.upper = descConv.dimInput;
    descConv.dimOutput = descConv.dimInput;
    descConv.learnbias = true;

    TestLayer *input = new TestLayer(nullptr);
    input->setRightVectorSize(4 * 8 * descConv.inChannel);
    input->setSize(1);
    input->prepare();

    SConvolutionalLayer *clayer = new SConvolutionalLayer(descConv, input);

    TestLayer *output = new TestLayer(clayer);
    output->setSize(1);
    output->prepare();
    output->setLeftVectorSize(4 * 8 * descConv.outChannel);

    clayer->init();
    clayer->prepare();
    clayer->displayKernel();
    clayer->displayBias();

    auto Xflat = RandomDevice::createUniformVector(4 * 8 * descConv.inChannel, -1.0, 1.0);
    auto dYflat = RandomDevice::createUniformVector(4 * 8 * descConv.outChannel, -1.0, 1.0);
    auto Xmask = RandomDevice::createMask(4 * 8 * descConv.inChannel, 0.5);
    auto dYmask = RandomDevice::createMask(4 * 8 * descConv.outChannel, 0.5);
    auto X = applyMask(Xflat, Xmask);
    auto dY = applyMask(dYflat, dYmask);
    auto Xfull = makeDense(X, Xmask, 4 * 8 * descConv.inChannel);
    auto dYfull = makeDense(dY, dYmask, 4 * 8 * descConv.outChannel);

    input->setOutput(0, X, Xmask);
    input->displayOutput(0);
    input->displayActiveOutput(0);

    output->setExpectedInput(0, {  // output channel 1
            -0.123288357609995, 0.301980589605635, 0.257515408865661, -0.252964360036999,
            -0.072648975440323, -0.019069531960598, 0.122421059316258, 0.049027841006028,
            -0.137100584794402, 0.122141231947358, -0.444182061307059, 0.269450105356494,
            -0.031420716073184, 0.169521363661365, -0.123521024551195, 0.268446476832641,
            0.173114571413652, -0.302006955219978, 0.186665872078588, -0.169137324260009,
            -0.103792846627348, -0.044812199440326, -0.090096811081646, 0.316074722196333,
            -0.077217422756947, 0.290034250503272, 0.134147590325396, -0.009158826330498,
            0.141887042079763, -0.153661986278561, 0.105126874188901, -0.157266774512773,

            // output channel 2
            -0.145775079437104, 0.162836948569692, 0.218826702851702, 0.033497236387529,
            -0.204942167529217, -0.074337451427945, -0.587823640732119, -0.081319403199409,
            -0.297426601222791, 0.356020099775463, 0.116144197852660, -0.025520459298736,
            0.177805758060709, 0.292819839881851, -0.282443465117338, 0.326870545062853,
            -0.344700423081608, -0.086010480245144, 0.131255945511349, 0.029630578603508,
            0.137284937431877, 0.109765211510298, -0.150359939544978, -0.047897007310655,
            0.027328632785853, -0.041415050964087, 0.138423933389230, 0.002306297778949,
            -0.021093899041268, -0.172831055467036, 0.105200083573140, -0.014331608083826
    }, {});

    output->setErrorSignal(0, dY, dYmask);

    output->displayLeftErrorSignal(0);
    output->displayActiveLeftErrorSignal(0);

    input->setExpectedErrorSignal(0, {
                                      -0.151554742715751,  -0.205944835642832,   0.506659306248191,   0.004759861677962,
                                      -0.220693393305775,  -0.005631013991776,   0.156354543052091,   0.195613734078516,
                                       0.186512083248380,  -0.152951358010863,  -0.074832768604702,   0.010722330386575,
                                       0.132484793603705,  -0.235751301898224,   0.194933435559029,   0.179884035530708,
                                       0.074635824900270,  -0.191003089521207,   0.139465545162946,   0.120719579188970,
                                       0.209628604267465,   0.151134995787154,  -0.357111696943550,  -0.094262727460613,
                                       0.092363030995825,   0.128346879939679,   0.127233687763584,   0.134459416616024,
                                      -0.084374578320445,  -0.186297911841305,   0.417651255371811,  -0.205001403456405,

                                      0.228603407468675,  0.335175895458417,  -0.366447504909689,    0.031715013551660,
                                      0.297733977044163,   0.182330668385574,   0.139027274252084,  -0.145078612054845,
                                     -0.182149090893768,   0.270032172211668,   0.087434570854510,   0.017464402827552,
                                     -0.039984191971127,   0.382568696445579,  -0.067961211232211,  -0.018212930753378,
                                     -0.047395617601611,   0.228324267549111,   0.028834438047118,  -0.019148410953067,
                                     -0.219665530251637,  -0.133752552135784,   0.326453941369462,  -0.007505466054360,
                                      0.051507150270261,  -0.027658822039395,  -0.048235187591279,   0.006185675284775,
                                      0.130915915706233,   0.131952780771789,  -0.409241094624930,   0.151286549583434}, {});

    clayer->prepare();
    clayer->feedforward();
    bool chkFeedForward = output->controlInput();
    if (!chkFeedForward) { sDebug() << "Feedforward not passed"; }

    clayer->backprop();
    bool chkKernelChanges = checkKernel({{
                                             -0.063459673755057,  -0.765778092988328,   0.311154342281438,
                                              1.115480303229967,   0.133220091202394,  -0.583030335937704,
                                              0.750043351520524,  -0.606006154505436,  -0.489610295808675


                                         },
                                         {        -0.834028904495300,   0.662074090513440,  -0.164457064401442,
                                                  -0.623274689388478,  -1.437327975587579,  -0.907837028229880,
                                                  -0.758273139101012,  -1.314919159457488,  -0.558709137974979
                                         }
                                        }, clayer);
    if (!chkKernelChanges) { sDebug() << "Gradient calculation not passed (Kernel)"; }

    bool chkBiasChanges = checkBias({-5.002671010109177}, clayer);
    if (!chkBiasChanges) { sDebug() << "Gradient calculation not passed (Bias)"; }

    bool chkBackProp = input->controlErrorSignal();
    if (!chkBackProp) { sDebug() << "Backpropagation not passed"; }

    delete input;
    delete clayer;
    delete output;
    return chkFeedForward && chkKernelChanges && chkBiasChanges && chkBackProp;
}

