#include "fullyconnectedlayer.h"

#include <leakyreluneuron.h> // todo
#include <softmaxneuron.h>


FullyConnectedLayer::FullyConnectedLayer() : AbstractLayer(){
    sDebug() << "Neither description nor previous layer provided for Activation Layer";
}

FullyConnectedLayer::FullyConnectedLayer(AbstractLayer *prev) : AbstractLayer(prev){
    sDebug() << "No description provided for Activation Layer";
}

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedDescription desc, AbstractLayer *prev): AbstractLayer(prev){
    this->type = LayerType::FullyConnected;
    this->leftActive = prev->rightActive;
    this->rightActive = new SharedActivation();
}

FullyConnectedLayer::FullyConnectedLayer( std::istream &stream, AbstractLayer * prev): AbstractLayer(stream, prev){
    this->type = LayerType::FullyConnected;
       FullyConnectedDescription desc;
       stream >> desc.szLeft;
       stream >> desc.szRight;

       qDebug() << "Connections "<< desc.szLeft << " x " << desc.szRight << sEndL();
       number tmpnumber;
       connections.resize(desc.szLeft);
       for(size_t pc = 0; pc < desc.szLeft; pc++){
           connections[pc].resize(desc.szRight);
           for(size_t tc = 0; tc < desc.szRight; tc++){
               stream >> tmpnumber;
               connections[pc][tc].weight =  tmpnumber;
           }
       }
}

FullyConnectedLayer::~FullyConnectedLayer(){

}

void FullyConnectedLayer::init(){
    LeakyReLUNeuron neuron; // todo weight initializer method ?
    size_t leftSize = this->leftActive->fullsize;
    size_t rightSize = this->rightActive->fullsize;

    connections.resize(leftSize);
    for(size_t i = 0; i < leftSize;i++){
        connections[i].resize(rightSize);
        for(size_t k=0; k< rightSize; k++){
            connections[i][k].weight = neuron.randomWeight(leftSize);
        }
    }

}


void FullyConnectedLayer::prepare(){
    this->size = this->previousLayer->size;
    this->data.resize(this->size);
}

void FullyConnectedLayer::feedforward(){
    //this->driver.displayWeights();

    size_t szOutputIndex = this->rightActive->active.size();
    if(szOutputIndex == 0){ // output should be dense
        size_t szOutpullFullSize = this->rightActive->fullsize;

        for(size_t bz = 0; bz < this->size; ++bz){
            auto & input = this->getInput(bz);

            auto &idxInput = this->getActiveInput(bz);
            size_t szInputIndex = idxInput.size();

            auto &output = this->getOutput(bz);
            auto &idxOutput = this->getActiveOutput(bz);

            output.resize(szOutpullFullSize);
            idxOutput.clear();


            if(szInputIndex == 0){ // dense Output & dense Input
                //sDebug() << "\t full input ";
                size_t szInputFullSize = this->leftActive->fullsize;
                for(size_t ra = 0; ra < szOutpullFullSize; ++ra){
                    number sum = 0.0;
                    for (size_t la = 0; la < szInputFullSize; ++la) {
                        sum += input[la] * connections[la][ra].weight;
                    }
                    output[ra] = sum;
                }
            } else { // Input sparse & Output dense
                //sDebug() << "\t sparse input";
                for(size_t ra = 0; ra < szOutpullFullSize; ++ra){
                    number sum = 0.0;
                    for (size_t la = 0; la < szInputIndex; ++la) {
                        sum += input[la] * connections[idxInput[la]][ra].weight;
                    }
                    output[ra] = sum;
                }
            }
        }
    } else { // output uses dropout
        for(size_t bz = 0; bz < this->size; ++bz){
            //sDebug() << "FC::feedforward sparse output vector ";
            auto & input = this->getInput(bz);
            auto & idxInput = this->getActiveInput(bz);
            auto szInputIndex = idxInput.size();

            auto &output = this->getOutput(bz);
            auto &idxOutput = this->getActiveOutput(bz);
            output.resize(szOutputIndex);
            idxOutput.resize(szOutputIndex);

            if(szInputIndex == 0){ // Input dense & Output sparse
                size_t szLeftFull = this->leftActive->fullsize;
                //sDebug() << "\t dense input ";
                for(size_t ra = 0; ra < szOutputIndex; ++ra){
                    idxOutput[ra] = this->rightActive->active[ra];
                    number sum = 0.0;
                    for (size_t la = 0; la < szLeftFull; ++la) {
                        sum += input[la] * connections[la][idxOutput[ra]].weight;
                    }
                    output[ra] = sum;
                }
            } else { // Input sparse & Output sparse
                //sDebug() << "\t sparse input ";
                for(size_t ra = 0; ra < szOutputIndex; ++ra){
                    idxOutput[ra] = this->rightActive->active[ra];
                    number sum = 0.0;
                    for (size_t la = 0; la < szInputIndex; ++la) {
                        sum += input[la]  * connections[idxInput[la]][idxOutput[ra]].weight;
                    }
                    output[ra] = sum;
                }
            }
        }
    }

}


void FullyConnectedLayer::calcInputChanges(){
    // dX_{r} = W_{l,r}^T * dX_{l}

    for(size_t bz = 0; bz < this->size; ++bz){
        auto & leftErrorSignal = this->getLeftErrorSignal(bz);
        auto & idxLeftErr = this->getActiveLeftErrorSignal(bz);

        auto & rightErrorSignal = this->getRightErrorSignal(bz);
        auto & idxRightErr = this->getActiveRightErrorSignal(bz);

        size_t leftActive = this->leftActive->active.size();
        size_t rightActive = idxRightErr.size();

        if(leftActive == 0){ // dense err
            size_t szLeftErrFullSize = this->leftActive->fullsize;
            idxLeftErr.clear();
            leftErrorSignal.resize(szLeftErrFullSize);

            if(rightActive == 0){ // dense err, sparse err
                size_t szRightFull = this->rightActive->fullsize;

                for(size_t la = 0 ; la < szLeftErrFullSize; ++la){
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightFull; ++ra) {
                        sum += connections[la][ra].weight * rightErrorSignal[ra];
                    }
                leftErrorSignal[la] = sum;
                }

            } else {
                for(size_t la = 0 ; la < szLeftErrFullSize; ++la){
                    number sum = 0.0;
                    for (size_t ra = 0; ra < rightActive; ++ra) {
                        sum += connections[la][idxRightErr[ra]].weight * rightErrorSignal[ra];
                    }
                leftErrorSignal[la] = sum;
                }
            }

        } else { // sparse err
            idxLeftErr.resize(leftActive);
            leftErrorSignal.resize(leftActive);

            if(rightActive == 0){ // sparse err, dense err
                size_t szRightFull = this->rightActive->fullsize;

                for(size_t la = 0 ; la < leftActive; ++la){
                    idxLeftErr[la] = this->leftActive->active[la];//
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightFull; ++ra) {
                        sum += connections[idxLeftErr[la]][ra].weight * rightErrorSignal[ra];
                    }

                leftErrorSignal[la] = sum;
                }


            } else {

                for(size_t la = 0 ; la < leftActive; ++la){
                    idxLeftErr[la] = this->leftActive->active[la];
                    number sum = 0.0;
                    for (size_t ra = 0; ra < rightActive; ++ra) {
                         sum += connections[idxLeftErr[la]][idxRightErr[ra]].weight * rightErrorSignal[ra];
                    }

                    leftErrorSignal[la] = sum;

                }

            }
        }
    }
}

void FullyConnectedLayer::calcWeightChanges(){
    // Y_{I(r)} = W_{I(r), I(l)}*X_{I(l)}
    // dW_{r,l} = dY_{r}^T * X_{l}

    for(size_t bz = 0; bz < this->size; ++bz){
        auto & rightErrorSignal = this->getRightErrorSignal(bz);
        auto & idxRightErr = this->getActiveRightErrorSignal(bz);

        auto & input = this->getInput(bz);
        auto & idxInput = this->getActiveInput(bz);

        if(idxInput.size() == 0){
            size_t szLeftSize = this->leftActive->fullsize;

            if(idxRightErr.size() == 0){ // dense input, dense error
                size_t szRightFull = this->rightActive->fullsize;
                for(size_t la = 0 ; la < szLeftSize; ++la){
                    for (size_t ra = 0; ra < szRightFull; ++ra) {
                        connections[la][ra].gradient += input[la]*rightErrorSignal[ra];
                    }
                }

            } else { // dense input, sparse error
                size_t szRightErrActive = idxRightErr.size();
                for(size_t la = 0 ; la < szLeftSize; ++la){
                    for (size_t ra = 0; ra < szRightErrActive; ++ra) {
                        connections[la][idxRightErr[ra]].gradient += input[la]*rightErrorSignal[ra];
                    }
                }
            }
        } else {
            size_t szLeftErrActive = idxInput.size();

            if(idxRightErr.size() == 0){ // sparse input, dense error
                size_t szRightErrFull = this->rightActive->fullsize;

                for(size_t la = 0 ; la < szLeftErrActive; ++la){
                    for (size_t ra = 0; ra < szRightErrFull; ++ra) {
                            connections[idxInput[la]][ra].gradient += input[la]*rightErrorSignal[ra];
                    }
                }

            } else { // sparse input, sparse error
                size_t szRightErrActive = idxRightErr.size();

                for(size_t la = 0 ; la < szLeftErrActive; ++la){
                    for (size_t ra = 0; ra < szRightErrActive; ++ra) {
                            connections[idxInput[la]][idxRightErr[ra]].gradient += input[la]*rightErrorSignal[ra];
                    }
                }

            }
        }
    }

}


void FullyConnectedLayer::backprop(){
    if(calcLeftErrorSignal){
        calcInputChanges();
    }
    calcWeightChanges();
    /*
    // todo loop leftErr, rightErr, Input

    size_t szLeftErrActive = this->leftActive->active.size();

    if(szLeftErrActive == 0){ // no leftErr dropout
        size_t szLeftErrFullSize = this->leftActive->fullsize;

        for(size_t bz = 0; bz < this->size; ++bz){
            auto & input = this->getInput(bz);
            auto & idxInput = this->getActiveInput(bz);
            auto & output = this->getOutput(bz);
            auto & idxOutput = this->getActiveOutput(bz);
            auto & rightErrorSignal = this->getRightErrorSignal(bz);
            auto & idxRightErr = this->getActiveRightErrorSignal(bz);
            auto & leftErrorSignal = this->getLeftErrorSignal(bz);
            auto & idxLeftErr = this->getActiveLeftErrorSignal(bz);

            idxLeftErr.clear();
            leftErrorSignal.resize(szLeftErrFullSize);

            size_t szRightErrActive = idxRightErr.size();


            if(szRightErrActive == 0){ // dense input
                if(idxInput.size() == 0){
                size_t szRightFull = this->rightActive->fullsize;

                for(size_t la = 0 ; la < szLeftErrFullSize; ++la){
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightFull; ++ra) {
                        connections[la][ra].gradient += input[la]*rightErrorSignal[ra];
                        sum += connections[la][ra].weight * rightErrorSignal[ra];
                    }
                leftErrorSignal[la] = sum;
                }
                } else {




                }

            } else { // sparse right error signal dense left error signal
                for(size_t la = 0 ; la < szLeftErrFullSize; ++la){
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightErrActive; ++ra) {
                        connections[la][idxRightErr[ra]].gradient += input[la]*rightErrorSignal[ra];
                        sum += connections[la][idxRightErr[ra]].weight * rightErrorSignal[ra];
                    }
                leftErrorSignal[la] = sum;
                }
            }
        }

    } else { // leftErr dropout

        for(size_t bz = 0; bz < this->size; ++bz){
            auto & input = this->getInput(bz);
            auto & idxInput = this->getActiveInput(bz);

            auto & output = this->getOutput(bz);
            auto & idxOutput = this->getActiveOutput(bz);

            auto & rightErrorSignal = this->getRightErrorSignal(bz);
            auto & idxRightErr = this->getActiveRightErrorSignal(bz);

            auto & leftErrorSignal = this->getLeftErrorSignal(bz);
            auto & idxLeftErr = this->getActiveLeftErrorSignal(bz);

            idxLeftErr.resize(szLeftErrActive);
            leftErrorSignal.resize(szLeftErrActive);

            auto & idxLeftActive = this->leftActive->active;

            size_t szRightErrActive = idxRightErr.size();

            if(szRightErrActive == 0){ // checked for szLeftErrActive == fullsize
                size_t szRightErrFull = this->rightActive->fullsize;
                size_t lainp = 0;
                for(size_t la = 0 ; la < szLeftErrActive; ++la){
                    idxLeftErr[la] = idxLeftActive[la];
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightErrFull; ++ra) {
                        if(idxInput[lainp] == idxLeftActive[la]){
                            //sDebug() << "W_" << idxInput[lainp] <<","<<ra << " += X_" << lainp << " * Err_" << ra;
                            //sDebug() << this->driver.connections[idxInput[lainp]][ra].gradient <<  " += " << input[lainp] << " * " << rightErrorSignal[ra]  << " = " <<  input[lainp]*rightErrorSignal[ra];
                            connections[idxInput[lainp]][ra].gradient += input[lainp]*rightErrorSignal[ra];
                        }
                        //sDebug() << "dX_" << la << " += W_"<< idxLeftErr[la] << ", " << ra << " * dY_" << ra;
                        //sDebug() << sum << "+= "<< this->driver.connections[idxLeftErr[la]][ra].weight << " * " << rightErrorSignal[ra];
                        sum += connections[idxLeftErr[la]][ra].weight * rightErrorSignal[ra];
                    }
                    if(idxInput[lainp] == idxLeftActive[la]){
                        lainp++;
                    }
                leftErrorSignal[la] = sum;
                }
            } else {  // checked for szRightErrActive == fullsize
                size_t lainp = 0;
                for(size_t la = 0 ; la < szLeftErrActive; ++la){
                    idxLeftErr[la] = this->leftActive->active[la];
                    number sum = 0.0;
                    for (size_t ra = 0; ra < szRightErrActive; ++ra) {
                        if(idxInput[lainp] == idxLeftActive[la]){
                            //sDebug() << "dW_" << idxInput[lainp] <<","<< idxRightErr[ra]<< " = X_" << lainp << " * dY_" ;
                            connections[idxInput[lainp]][idxRightErr[ra]].gradient += input[lainp]*rightErrorSignal[ra];
                        }
                        sum += connections[idxLeftErr[la]][idxRightErr[ra]].weight * rightErrorSignal[ra];
                    }
                if(idxInput[lainp] == idxLeftActive[la]){
                     lainp++;
                }
                leftErrorSignal[la] = sum;
                }

            }
        }
    }*/
}

void FullyConnectedLayer::update(size_t epoch){
    auto & leftActiveIndex = this->leftActive->active;
    auto & rightActiveIndex = this->rightActive->active;

    size_t leftActive = leftActiveIndex.size();
    size_t rightActive = rightActiveIndex.size();


    if(leftActive == 0){ // left all active
        size_t szLeftFull = this->leftActive->fullsize;

        for(size_t la = 0; la < szLeftFull; ++la){
            size_t szRightActive = this->rightActive->active.size();
            if(szRightActive == 0){
                size_t szRightFull = this->rightActive->fullsize;
                for(size_t ra = 0; ra < szRightFull; ++ra){
                    auto & conn = connections[la][ra];
                    updater.update(conn,epoch);
                }
            } else {
                for(size_t ra = 0; ra < szRightActive; ++ra){
                    auto & conn = connections[la][this->rightActive->active[ra]];
                    updater.update(conn,epoch);
                }
            }
        }
    } else {
        for(size_t la = 0; la < leftActive; ++la){
            if(rightActive == 0){
                size_t szRightFull = this->rightActive->fullsize;
                for(size_t ra = 0; ra < szRightFull; ++ra){
                    auto & conn = connections[leftActiveIndex[la]][ra];
                    updater.update(conn,epoch);
                }
            } else {
                for(size_t ra = 0; ra < rightActive; ++ra){
                    auto & conn = connections[leftActiveIndex[la]][this->rightActive->active[ra]];
                    updater.update(conn,epoch);
                }
            }
        }
    }
}

void FullyConnectedLayer::serialize(std::ostream &out){
    const char * sendl = "\n";
    size_t szLeft = leftActive->fullsize;
    size_t szRight = rightActive->fullsize;

    out << szLeft << " " << szRight << sendl;
    for( auto & row : connections){
        for(auto & connection : row){
            out << connection.weight << " ";
        }
        out << sendl;
    }
    out << sendl;
}


void FullyConnectedLayer::displayOutput(){
    for(size_t bz = 0; bz < this->size; bz++){
        sDebug() << " bz  " << bz;
        for(size_t i = 0; i < this->data[bz].output.size(); i++){
            sDebug() << this->data[bz].activeOutput[i] << " "<<  this->data[bz].output[i];
        }
    }
}

void FullyConnectedLayer::displayWeights()
{
    const size_t szPrev = this->connections.size();
    const size_t szNext = this->connections[0].size();
    for(size_t from = 0; from < szPrev; ++from){
        auto & conn = this->connections[from];
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        for(size_t to = 0; to < szNext; ++to){
            ss << conn[to].weight << " ";
        }
        sDebug() << ss.str().c_str();
    }
}

void FullyConnectedLayer::displayWeightChanges()
{
    const size_t szPrev = this->connections.size();
    const size_t szNext = this->connections[0].size();
    for(size_t from = 0; from < szPrev; ++from){
        auto & conn = this->connections[from];
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        for(size_t to = 0; to < szNext; ++to){
            ss << conn[to].gradient << " ";
        }
        sDebug() << ss.str().c_str();
    }
}
