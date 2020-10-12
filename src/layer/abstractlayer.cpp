#include "abstractlayer.h"
#include "inputlayer.h"

AbstractLayer::AbstractLayer(){

}

AbstractLayer::AbstractLayer(AbstractLayer *prev){
    this->previousLayer = prev;
    if(prev){
        previousLayer->nextLayer = this;
        InputLayer* inLayer = dynamic_cast<InputLayer * >(prev);
        if(inLayer){
            this->calcLeftErrorSignal = false;
        }
    }
}

AbstractLayer::AbstractLayer(std::istream &stream, AbstractLayer * prev){
        // todo
}

AbstractLayer::~AbstractLayer(){
    // todo
}

void AbstractLayer::setTraining(bool training){
    this->training = training;
}

void AbstractLayer::displayOutput(size_t index){
    sDebug() << " =============== Output =============== ";
    auto & val = this->getOutput(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayActiveOutput(size_t index)
{
    sDebug() << " =============== Active Output =============== ";
    auto & val = this->getActiveOutput(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayInput(size_t index)
{
    sDebug() << " =============== Input =============== ";
    auto & val = this->getInput(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayActiveInput(size_t index)
{
    sDebug() << " =============== Active Input =============== ";
    auto & val = this->getActiveInput(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayRightErrorSignal(size_t index)
{
    sDebug() << " =============== Right Error Signal =============== ";
    auto & val = this->getRightErrorSignal(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayActiveRightErrorSignal(size_t index)
{
    sDebug() << " =============== Active Right Error Signal =============== ";
    auto & val = this->getActiveRightErrorSignal(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

void AbstractLayer::displayLeftErrorSignal(size_t index){
    sDebug() << " =============== Left Error Signal =============== ";
auto & val = this->getLeftErrorSignal(index);
for(size_t i = 0; i < val.size(); i++){
    std::stringstream ss;
    ss << std::setprecision(WRITE_PRECISION);
    ss << val[i];
    sDebug() <<ss.str().c_str();
}
}

void AbstractLayer::displayActiveLeftErrorSignal(size_t index)
{
    sDebug() << " =============== Active Left Error Signal =============== ";
    auto & val = this->getActiveLeftErrorSignal(index);
    for(size_t i = 0; i < val.size(); i++){
        std::stringstream ss;
        ss << std::setprecision(WRITE_PRECISION);
        ss << val[i];
        sDebug() <<ss.str().c_str();
    }
}

