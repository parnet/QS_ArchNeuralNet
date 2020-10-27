#include <QApplication>
#include <QtGlobal>
#include <neuralnet.h>
#include <qgprawdata.h>
#include <qgpsequentialbatch.h>

#include "environment.h"
#include "randomdevice.h"
#include "ui/mainwindow.h"
#include "../test/test.cpp"

#include "logging.h"

int devconvert(int argc, char*argv[]){
    Module * module = new Module();
    //module->fileDimension = Dimension(4,20,20,20,14); // Azimut (-pi, pi) , Inclination ( 0, pi), momentum, particlecount
    module->filename.path = "M:/sphr_28_20_20_20/";

    QGPRawData qgpr{};
    qgpr.nFilesPerClass = 5;
    qgpr.inFilepath = "Q:/QS_NeuralNet/data_raw/31.2/";

    qgpr.convert(module);
    exit(-20);
}

int devmain(int argc, char*argv[]){

    Module * module = new Module();
    std::vector<size_t> dimInput = {20,20,20,28};
    module->fileDimension = Dimension(dimInput);
    module->filename.path = "M:/sphr_28_20_20_20";

    NeuralNet net = NeuralNet();
    QGPBatch * batch = new QGPSequentialBatch(module);

    batch->fromIndex = 0;
    batch->toIndex = 10;
    batch->size = 20;

    batch->load();

    net.prepare(batch,true);
    exit(-20);
}

int devfc(int argc, char *argv[]){
    std::vector<size_t> dimInput = {20, 20, 1, 1};
    Topology top;

    InputLayerDescription descIn;
    descIn.dimension = Dimension(dimInput);
    descIn.size = descIn.size;
    top.addDescription(descIn);

    FullyConnectedDescription descFCA;
    descFCA.szLeft = descIn.size;
    descFCA.szRight = 20;
    top.addDescription(descFCA);

    ActivationLayerDescription descActA;
    descActA.dropout = 0.5;
    descActA.usesbias = true;
    descActA.activation = NeuronType::ReLU;
    descActA.numberOfNeurons = 20;
    top.addDescription(descActA);

    FullyConnectedDescription descFCB;
    descFCB.szLeft = descActA.numberOfNeurons;
    descFCB.szRight = 2;
    top.addDescription(descFCB);

    ActivationLayerDescription descActB;
    descActB.dropout = 0.0;
    descActB.usesbias = false;
    descActB.activation = NeuronType::Softmax;
    descActB.numberOfNeurons = 2;
    top.addDescription(descActB);

    NeuralNet * net = new NeuralNet(top);



    exit(2);
}

int main(int argc, char *argv[]) {
    sDebug() << "num of args: " << argc << qSetRealNumberPrecision(2);
    RandomDevice::setSeed(1024);  // todo delete seed or notice user
    //qInstallMessageHandler(&loggingFile);
    //__test();
    // devconvert(argc,argv);
    // devmain(argc, argv);
    // devfc(argc, argv);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
