#-------------------------------------------------
#
# Project created by QtCreator 2019-10-01T15:00:00
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = QS_ArchNeuralNet
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

CONFIG += c++17

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH += libs/ \
               libs/qcustomplot/ \
               src/ \
               src/ui/ \
               src/update/ \
               src/neuron/ \
               src/layer/ \
               src/sparse/ \
               src/description/ \
               src/convert/ \
               src/data/ \
               src/net/ \
               test/ \

SOURCES += \
    libs/qcustomplot/qcustomplot.cpp \
    src/convert/coordinatetransformation.cpp \
    src/convert/indexprovider.cpp \
    src/convert/qgprawdata.cpp \
    src/data/qgpbatch.cpp \
    src/data/qgpdata.cpp \
    src/data/qgpgriddata.cpp \
    src/data/qgpparticledata.cpp \
    src/data/qgprandombatch.cpp \
    src/data/qgpsequentialbatch.cpp \
    src/data/qgpsparsedata.cpp \
    src/data/qgpsparsestoreddata.cpp \
    src/data/qgpstatistics.cpp \
    src/dimension.cpp \
    src/layer/abstractlayer.cpp \
    src/layer/activationlayer.cpp \
    src/layer/averagepoolinglayer.cpp \
    src/layer/combinedlayer.cpp \
    src/layer/convolutionallayer.cpp \
    src/layer/euklidianpoolinglayer.cpp \
    src/layer/fullyconnectedlayer.cpp \
    src/layer/generaldata.cpp \
    src/layer/inputdata.cpp \
    src/layer/inputlayer.cpp \
    src/layer/layerfactory.cpp \
    src/layer/lppoolinglayer.cpp \
    src/layer/maxpoolingdata.cpp \
    src/layer/maxpoolinglayer.cpp \
    src/layer/normalizationdata.cpp \
    src/layer/normalizationdriver.cpp \
    src/layer/normalizationlayer.cpp \
    src/layer/outputdata.cpp \
    src/layer/outputlayer.cpp \
    src/layer/poolingdata.cpp \
    src/layer/sconvolutionallayer.cpp \
    src/layer/selectiveconvolutionlayer.cpp \
    src/layer/sharedactivation.cpp \
    src/layer/sparseconnectedlayer.cpp \
    src/layer/testlayer.cpp \
    src/neuron/abstractdropoutfunction.cpp \
    src/sparse/sparseabstractlayer.cpp \
    src/sparse/sparseconvolutionaldriver.cpp \
    src/sparse/sparseconvolutionallayer.cpp \
    src/sparse/sparsedata.cpp \
    src/layer/stochasticpoolinglayer.cpp \
    src/main.cpp \
    src/module.cpp \
    src/net/neuralnet.cpp \
    src/net/supervisor.cpp \
    src/net/topology.cpp \
    src/neuron/abstractneuron.cpp \
    src/neuron/biasneuron.cpp \
    src/neuron/identityneuron.cpp \
    src/neuron/leakyreluneuron.cpp \
    src/neuron/logisticneuron.cpp \
    src/neuron/reluneuron.cpp \
    src/neuron/softmaxagineuron.cpp \
    src/neuron/softmaxdiagonalneuron.cpp \
    src/neuron/softmaxneuron.cpp \
    src/neuron/tanhneuron.cpp \
    src/randomdevice.cpp \
    src/sparse/sparsemaxpooling.cpp \
    src/sparse/sparsepoolingdata.cpp \
    src/ui/choosewidget.cpp \
    src/ui/convertwidget.cpp \
    src/ui/mainwindow.cpp \
    src/ui/topologywidget.cpp \
    src/ui/trainingwidget.cpp \
    src/ui/validationwidget.cpp \
    test/layer/testconvolutionlayer.cpp \
    test/layer/testfullyconnectedlayer.cpp \
    test/layer/testmaxpoolinglayer.cpp \
    test/layer/testsconvolutionlayer.cpp \
    test/test.cpp \
    test/testutil.cpp \



HEADERS += \
    libs/qcustomplot/qcustomplot.h \
    src/convert/coordinatetransformation.h \
    src/convert/indexprovider.h \
    src/convert/qgprawdata.h \
    src/data/filename.h \
    src/data/qgpbatch.h \
    src/data/qgpclassification.h \
    src/data/qgpdata.h \
    src/data/qgpdimension.h \
    src/data/qgpgriddata.h \
    src/data/qgpparticledata.h \
    src/data/qgprandombatch.h \
    src/data/qgpsequentialbatch.h \
    src/data/qgpsparsedata.h \
    src/data/qgpsparsestoreddata.h \
    src/data/qgpstatistics.h \
    src/dimension.h \
    src/environment.h \
    src/layer/abstractlayer.h \
    src/description/abstractlayerdescription.h \
    src/layer/activationlayer.h \
    src/description/activationlayerdescription.h \
    src/layer/averagepoolinglayer.h \
    src/layer/combinedlayer.h \
    src/layer/convolutionallayer.h \
    src/description/convolutionlayerdescription.h \
    src/layer/euklidianpoolinglayer.h \
    src/description/fullyconnecteddescription.h \
    src/layer/fullyconnectedlayer.h \
    src/layer/generaldata.h \
    src/layer/inputdata.h \
    src/layer/inputlayer.h \
    src/description/inputlayerdescription.h \
    src/layer/layerfactory.h \
    src/layer/layertype.h \
    src/layer/lppoolinglayer.h \
    src/layer/maxpoolingdata.h \
    src/layer/maxpoolinglayer.h \
    src/description/maxpoolinglayerdescription.h \
    src/layer/normalizationdata.h \
    src/layer/normalizationdriver.h \
    src/layer/normalizationlayer.h \
    src/layer/outputdata.h \
    src/layer/outputlayer.h \
    src/description/outputlayerdescription.h \
    src/layer/paddingtype.h \
    src/layer/poolingdata.h \
    src/layer/sconvolutionallayer.h \
    src/layer/selectiveconvolutionlayer.h \
    src/layer/sharedactivation.h \
    src/layer/sparseconnectedlayer.h \
    src/layer/testlayer.h \
    src/neuron/abstractdropoutfunction.h \
    src/sparse/sparseabstractlayer.h \
    src/sparse/sparseconvolutionaldriver.h \
    src/sparse/sparseconvolutionallayer.h \
    src/sparse/sparsedata.h \
    src/layer/stochasticpoolinglayer.h \
    src/logging.h \
    src/module.h \
    src/net/neuralnet.h \
    src/net/supervisor.h \
    src/net/topology.h \
    src/neuron/abstractneuron.h \
    src/neuron/biasneuron.h \
    src/neuron/identityneuron.h \
    src/neuron/leakyreluneuron.h \
    src/neuron/logisticneuron.h \
    src/neuron/neuronfactory.h \
    src/neuron/neurontype.h \
    src/neuron/reluneuron.h \
    src/neuron/softmaxagineuron.h \
    src/neuron/softmaxdiagonalneuron.h \
    src/neuron/softmaxneuron.h \
    src/neuron/tanhneuron.h \
    src/randomdevice.h \
    src/sparse/sparsemaxpooling.h \
    src/sparse/sparsepoolingdata.h \
    src/ui/choosewidget.h \
    src/ui/convertwidget.h \
    src/ui/topologywidget.h \
    src/ui/trainingwidget.h \
    src/ui/validationwidget.h \
    src/update/weightadamoptimization.h \
    src/update/weightgradientdescent.h \
    src/update/weightmomentumofinertia.h \
    src/ui/mainwindow.h \
    test/layer/testconvolutionlayer.h \
    test/layer/testfullyconnectedlayer.h \
    test/layer/testmaxpoolinglayer.h \
    test/layer/testsconvolutionlayer.h \
    test/testutil.h \


 FORMS += \
    src/ui/choosewidget.ui \
    src/ui/convertwidget.ui \
    src/ui/mainwindow.ui \
    src/ui/topologywidget.ui \
    src/ui/trainingwidget.ui \
    src/ui/validationwidget.ui \

ICON = res/qgp.ico
win32:RC_ICONS += res/qgp.ico

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_LFLAGS_RELEASE -= -O1

QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -fopenmp

DISTFILES += \
    abernanit.txt \
    abernanit.txt
