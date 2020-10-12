#include "activationlayer.h"
#include "neurontype.h"
#include "neuronfactory.h"

ActivationLayer::ActivationLayer() : AbstractLayer(){
    sDebug() << "Neither description nor previous layer provided for Activation Layer";
}

ActivationLayer::ActivationLayer(AbstractLayer *prev) : AbstractLayer(prev){
    sDebug() << "No description provided for Activation Layer";
}

ActivationLayer::ActivationLayer(ActivationLayerDescription desc, AbstractLayer *prev) : AbstractLayer(prev){
    this->type = LayerType::Activation;
    this->desc = desc;

    this->leftActive = prev->rightActive;
    this->rightActive = this->leftActive;

    this->leftActive->fullsize = desc.numberOfNeurons;

    if(desc.usesbias){
        this->bias.resize(desc.numberOfNeurons);
    }

    this->neurons.resize(desc.numberOfNeurons);
    for(size_t i = 0; i < desc.numberOfNeurons; i++){
        this->neurons[i] = NeuronFactory::create(desc.activation);
    }
}

ActivationLayer::ActivationLayer(std::istream &stream, AbstractLayer * prev) : AbstractLayer(prev){
    ActivationLayerDescription desc;
    desc.dropout;
    desc.activation;
    desc.numberOfNeurons;
    desc.usesbias;
}

ActivationLayer::~ActivationLayer()
{
    // todo
    /*const size_t szNeurons = this->neurons.size();
    for(size_t n = 0; n < szNeurons; ++n){
        delete this->neurons[n];
    }
    this->neurons.resize(0);*/
}

void ActivationLayer::init(){
    // todo
}

void ActivationLayer::prepare(){
    this->size = previousLayer->size;
    this->data.resize(previousLayer->size);
    this->leftActive->active.clear();
    if(desc.dropout != 0.0 && training){
        for(size_t i = 0; i < this->desc.numberOfNeurons; i++){
            if(this->neurons[i]->dropout(desc.dropout)){
                this->leftActive->active.emplace_back(i);
            }
        }
        this->outputscaling = 1.0 / (1-desc.dropout);
        if(this->leftActive->active.size() == 0){
            sDebug() << "everything dropped";
        }
    } else {
        this->outputscaling = 1.0;
        /*for(size_t i =0 ; i < this->leftActive->fullsize; i++){
            this->leftActive->active.push_back(i);
        }*/
    }
    /*size_t epoch = localepoch / 80 + 1;
    localepoch++;

    if (training && this->desc.dropout != 0.0) {
        number prob = pow(0.7, (0.5 * epoch + 0.12)); // desc.dropout */
    //sDebug() << "active " << this->leftActive->active.size()<< "\t" << this->leftActive->active;
}

void ActivationLayer::feedforward(){
    for(size_t bz = 0; bz < this->size; bz++){
        auto & input = getInput(bz);
        auto & idxInput = getActiveInput(bz);
        auto & output = getOutput(bz);
        auto & idxOutput = getActiveOutput(bz);

        size_t szActive = idxInput.size();

        if(szActive == 0){ // dense input, dense output
            size_t szActiveFull = leftActive->fullsize;
            output.resize(szActiveFull);
            idxOutput.clear();

            if(this->desc.usesbias){
                for(size_t a = 0; a < szActiveFull; ++a){
                    input[a] += bias[a].weight;
                }
            }
            for(size_t a = 0; a < szActiveFull; ++a){
                output[a] = this->neurons[a]->activate(a,input) * this->outputscaling ;
            }
        } else { // sparse input, sparse output
            //idxOutput = idxInput;
            output.resize(szActive);
            idxOutput.resize(szActive);

            if(this->desc.usesbias){
                for(size_t a = 0; a < szActive; ++a){
                    input[a] += bias[idxInput[a]].weight;
                }
            }
            output.resize(szActive);
            for(size_t a = 0; a < szActive; ++a){
                idxOutput[a] = idxInput[a];
                output[a] = this->neurons[idxInput[a]]->activate(a,input) * this->outputscaling ;
            }
        }
    }
}

void ActivationLayer::backprop(){

    //auto & activeIndex = this->rightActive->active;
    //size_t szActive = activeIndex.size();



    for(size_t bz = 0; bz < this->size; ++bz){

        auto &rightErrorSignal = this->getRightErrorSignal(bz);
        auto &activeRightErrorSignal = this->getActiveRightErrorSignal(bz);
        auto &activeErrorSignal = this->getActiveLeftErrorSignal(bz);
        size_t szActive = activeRightErrorSignal.size();

        //auto &activeOutput = this->getActiveOutput(bz);


        auto &input = this->getInput(bz);
        auto &leftErrorSignal = data[bz].errorSignal;
        //qDebug() << activeRightErrorSignal;
        //qDebug() << this->data[bz].activeErrorSignal.size();
        //this->data[bz].activeErrorSignal = this->leftActive->active;
        //activeOutput = activeRightErrorSignal;


        if(szActive == 0){
            size_t szActive = this->rightActive->fullsize;
            leftErrorSignal.resize(szActive);
            activeErrorSignal.clear();
            for(size_t a = 0; a < szActive; a++){
                leftErrorSignal[a] = neurons[a]->backpass(a,rightErrorSignal,input);
            }

            if(desc.usesbias){
                for(size_t a = 0; a < szActive; a++){
                    bias[a].gradient += rightErrorSignal[a];
                }
            }
        } else {
            leftErrorSignal.resize(szActive);
            activeErrorSignal.resize(szActive);
            for(size_t a = 0; a < szActive; a++){
                activeErrorSignal[a] = activeRightErrorSignal[a];
                leftErrorSignal[a] = neurons[activeRightErrorSignal[a]]->backpass(a,rightErrorSignal,input);
            }

            if(desc.usesbias){
                for(size_t a = 0; a < szActive; a++){
                    bias[activeRightErrorSignal[a]].gradient += rightErrorSignal[a];
                }
            }
        }


    }
    /*

    for(size_t bz = 0; bz < this->size; bz++){
          sDebug() << " ======================================================== " << bz;
          sDebug() << " output ";
          for(size_t i =0; i < this->data[bz].output.size(); i++){
              sDebug() << this->data[bz].output[i];
          }
           sDebug() << " ------------------------------------------------------- " << bz;
           sDebug() << " output index ";
           for(size_t i =0; i < this->data[bz].activeOutput.size(); i++){
               sDebug() << this->data[bz].activeOutput[i];
           }
            sDebug() << " ------------------------------------------------------- " << bz;
            sDebug() << " errorsignal ";
            for(size_t i =0; i < this->data[bz].errorSignal.size(); i++){
                sDebug() << this->data[bz].errorSignal[i];
            }
             sDebug() << " ------------------------------------------------------- " << bz;
             sDebug() << " output index ";
             for(size_t i =0; i < this->data[bz].activeErrorSignal.size(); i++){
                 sDebug() << this->data[bz].activeErrorSignal[i];
             }
             sDebug() << " ------------------------------------------------------- " << bz;
    }*/
}

void ActivationLayer::update(size_t epoch){
    // todo active
    for(size_t i = 0; i < this->neurons.size(); i++){
        this->neurons[i]->update(epoch);
    }
    if(desc.usesbias){
        for(size_t i = 0; i < this->neurons.size(); i++){
            biasUpdater.update(bias[i], epoch);
        }
    }
}

void ActivationLayer::serialize(std::ostream &out){
    // todo
/*

    out << szNeuronsWithoutBias << " "
        << dropoutChance << " "
        << hasBias << sendl;*/

}
