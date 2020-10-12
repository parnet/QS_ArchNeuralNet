#include "inputdata.h"


InputData::InputData(){

}

void InputData::setData(QGPData &data){
    // todo review
    const size_t szData = data.data.size();
    this->output.resize(szData);
    this->active.resize(szData);
    for(size_t i = 0; i < szData; i++){
        this->output[i] = data.data[i];
        this->active[i] = i;
    }
}

void InputData::setData(QGPSparseStoredData &data){
    // todo review
    const size_t szData = data.data.size();
    this->output.resize(szData);
    this->active.resize(szData);
    std::fill(&this->output[0], &this->output[szData],0);
    for(size_t i = 0; i < szData; i++){
        this->output[i] = data.data[i];
        this->active[i] = i;
    }
}

void InputData::setData(QGPSparseData &data){
    const size_t szData = data.data.size();
    this->active.resize(szData);
    this->output.resize(szData);
    for(size_t i = 0; i < szData; i++){
        this->active[i] = data.index[i];
        this->output[i] = data.data[i];
    }
}
