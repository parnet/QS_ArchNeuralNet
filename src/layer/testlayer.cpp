#include "testlayer.h"

TestLayer::TestLayer() : AbstractLayer(nullptr){
    this->leftActive = nullptr;
    this->rightActive = new SharedActivation();
}

TestLayer::TestLayer(AbstractLayer *prev) : AbstractLayer(prev){
    if(prev != nullptr){
        this->leftActive = prev->rightActive;
    }
    this->rightActive = new SharedActivation();
}

TestLayer::~TestLayer() {

}

void TestLayer::init(){}

void TestLayer::prepare(){
    this->data.resize(this->size);
    this->compare.resize(this->size);
}

void TestLayer::setSize(size_t size){
    this->size = size;
}

void TestLayer::setRightVectorSize(size_t size)
{
        this->rightActive->fullsize = size;
}

void TestLayer::setLeftVectorSize(size_t size)
{
    this->leftActive->fullsize = size;
}

void TestLayer::setLeftActive(std::vector<size_t> active){
    this->leftActive->active = active;
}

void TestLayer::setRightActive(std::vector<size_t> active){
    this->leftActive->active = active;
}

void TestLayer::setTraining(bool training){
    this->training = training;
}

void TestLayer::update(size_t epoch){}

void TestLayer::feedforward(){}

void TestLayer::backprop(){}

void TestLayer::serialize(std::ostream &out){}

bool TestLayer::controlInput(){
    bool passed = true;

    const double tol = 1e-5;

    for(size_t i = 0; i < this->size; i++){

        auto & input = this->getInput(i);
        auto & expected = this->compare[i].output;

        auto & idxInput = this->getActiveInput(i);
        auto & idxExpected = this->compare[i].activeOutput;

        // check storage size
        if(idxInput.size() == idxExpected.size()){
            for(size_t j = 0; j < idxInput.size(); j++){
                if(idxInput[j] != idxExpected[j]){
                    sDebug() << "index missmatch ("<<input[j] << ","<< expected[j]<<") for Input @"<<i <<" pos @" << j;
                    passed = false;
                }
            }
        } else {
            sDebug() << "size missmatch for input index @" << i;
            passed = false;
        }

        // check vector size
        if(input.size() == expected.size()){
            // check values
            for(size_t j = 0; j < input.size(); j++){
                if(fabs(input[j] - expected[j]) > 1e-5){
                    sDebug() << "dist("<<input[j] << ","<< expected[j]<<") too big for Input @"<<i <<" pos @" << j;
                    passed = false;
                }
            }
        } else {
            sDebug() << "size missmatch for Input @" << i;
            passed = false;
        }
    }
}

bool TestLayer::controlErrorSignal(){
    bool passed = true;
    const double tol = 1e-5;


    for(size_t i = 0; i < this->size; i++){

        auto & value = this->getRightErrorSignal(i);
        auto & expected = this->compare[i].errorSignal;

        auto & idxInput = this->getActiveRightErrorSignal(i);
        auto & idxExpected = this->compare[i].activeErrorSignal;

        // check storage size
        if(idxInput.size() == idxExpected.size()){
            for(size_t j = 0; j < idxInput.size(); j++){
                if(idxInput[j] != idxExpected[j]){
                    sDebug() << "index missmatch ("<<value[j] << ","<< expected[j]<<") for ErrorSignal @"<<i <<" pos @" << j;
                    passed = false;
                }
            }
        } else {
            sDebug() << "size missmatch for index ErrorSignal @" << i;
            passed = false;
        }

        // check vector size
        if(value.size() == expected.size()){
            // check values
            for(size_t j = 0; j < value.size(); j++){
                if(fabs(value[j] - expected[j]) > 1e-5){
                    sDebug() << "dist("<<value[j] << ","<< expected[j]<<") too big for ErrorSignal @"<<i <<" pos @" << j;
                    passed = false;
                }
            }
        } else {
            sDebug() << "size missmatch for ErrorSignal @" << i;
            passed = false;
        }
    }
    return passed;
}


void TestLayer::setOutput(size_t index, std::vector<number> value, std::vector<size_t> active){
    this->data[index].output = value;
    this->data[index].activeOutput = active;
}

void TestLayer::setExpectedInput(size_t index, std::vector<number> value, std::vector<size_t> active)
{
    this->compare[index].output = value;
    this->compare[index].activeOutput = active;
}

void TestLayer::setErrorSignal(size_t index, std::vector<number> value, std::vector<size_t> active){
    this->data[index].errorSignal = value;
    this->data[index].activeErrorSignal = active;
}

void TestLayer::setExpectedErrorSignal(size_t index, std::vector<number> value, std::vector<size_t> active){
    this->compare[index].errorSignal = value;
    this->compare[index].activeErrorSignal = active;
}

