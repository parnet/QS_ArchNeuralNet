#include "neuralnet.h"
#include <activationlayer.h>
#include <convolutionallayer.h>
#include <fstream>
#include <fullyconnectedlayer.h>
#include <layerfactory.h>
#include <maxpoolinglayer.h>
#include <normalizationlayer.h>

NeuralNet::NeuralNet(){}

NeuralNet::NeuralNet(Topology topology){
    apply(topology);
}

NeuralNet::~NeuralNet(){
    this->reset();
}

void NeuralNet::change(Topology &topology){
    this->reset();
    this->apply(topology);
}

void NeuralNet::reset(){
    const size_t numberOfLayers = this->layers.size();
    for (size_t k = 0; k < numberOfLayers; ++k) {
        delete this->layers[k];
    }
    this->layers.clear();
}

void NeuralNet::apply(Topology &topology){
    AbstractLayer *prevd = nullptr;
    const size_t numberOfLayers = topology.layers.size();
    for (size_t currentLayer = 0; currentLayer < numberOfLayers; currentLayer++) {
        if(currentLayer > 0){
            prevd = this->layers.back();
        }
        this->layers.push_back(LayerFactory::createLayer(topology.layers[currentLayer], prevd));
    }

    for(auto * layer : this->layers){
        layer->init();
    }
}

NeuralNet* NeuralNet::fromFile(std::string &filepath){
    NeuralNet * nn =new NeuralNet();

    std::ifstream file(filepath.c_str());
    // number of layers
    size_t szLayers;
    file >> szLayers;
    // layer types
    std::vector<LayerType> layerTypes;
    nn->layers.resize(szLayers);
    size_t tmpSize;
    for (size_t i = 0; i < szLayers; i++) {
        file >> tmpSize;
        auto layerType = static_cast<LayerType>(tmpSize);
        layerTypes.push_back(layerType);
    }

    FullyConnectedDescription desc;  // todo

    AbstractLayer * prev = nullptr; // todo
    for(size_t i = 0; i < szLayers; i++){
        AbstractLayer * cLayer = LayerFactory::createLayer(&desc,prev); // todo

        nn->layers[i] = cLayer;
        prev = cLayer;
    }
    return nn;
}

void NeuralNet::toFile(const std::string &filepath) const{
    std::ofstream file(filepath.c_str());
    file << std::setprecision(WRITE_PRECISION);
    // number of layers
    size_t szLayers = this->layers.size();
    file << szLayers << std::endl;
    // layer types
    for (size_t i = 0; i < szLayers; i++) {
        file << this->layers[i]->type << std::endl;
    }
    file << std::endl;

    for (auto & layer : this->layers) {
        layer->serialize(file);
        file << std::endl;
    }
    file.close();
}

void NeuralNet::prepare(QGPBatch *batch, bool training){
    for(auto & _layer : this->layers){
        _layer->setTraining(training);
    }
    auto * inputLayer = static_cast<InputLayer*>(this->layers[0]);
    auto * outputLayer = static_cast<OutputLayer*>(this->layers.back());
    inputLayer->setInput(batch);
    outputLayer->setTarget(batch);

	for(auto &_layer : this->layers){
		_layer->prepare();
	}
}

void NeuralNet::getResult(std::vector<number> &result, double & loss){
    auto* output = static_cast<OutputLayer*>(this->layers.back());
    result = output->getOutput(0); // todo 0?
    loss = output->getLoss(0);
}

void NeuralNet::backprop(QGPData *data, size_t epoch){
    // todo
}

void NeuralNet::backprop(QGPBatch *batch, size_t epoch){
    for(size_t i = layers.size()-1; i != 0; --i){
        layers[i]->backprop();
    }
    layers[0]->backprop();

    for(auto &_layer : layers){
        _layer->update(epoch);
    }
}

void NeuralNet::prepare(QGPData *data, bool training){
    // todo
}

void NeuralNet::feedForward(QGPData *data){
    // todo

}

void NeuralNet::getResults(std::vector<std::vector<number> > &results){
    // todo
}

void NeuralNet::getResults(std::vector<std::vector<number> > &results, std::vector<number> &loss){
    auto *outputLayer = static_cast<OutputLayer*>(this->layers.back());
    outputLayer->getTrainingResults(results,loss);
}

void NeuralNet::getResult(std::vector<number> &result){
    auto * outputLayer = this->layers.back();
    result = outputLayer->getOutput(0);
}


void NeuralNet::feedForward(QGPBatch *batch){
    const size_t szLayers = this->layers.size();
    InputLayer * inputLayer = static_cast<InputLayer*>(layers[0]);
    inputLayer->setInput(batch);

    for(auto & layer : this->layers){
        layer->feedforward();
    }
    //exit(-1);
}
