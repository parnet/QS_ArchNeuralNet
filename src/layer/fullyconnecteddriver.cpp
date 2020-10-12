#include "fullyconnecteddriver.h"

FullyConnectedDriver::FullyConnectedDriver(){

}

void FullyConnectedDriver::displayWeightChanges(){
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

void FullyConnectedDriver::displayWeights(){
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

void FullyConnectedDriver::update(size_t epoch){
    /*size_t szPActive = 0;
       if(this->previousLayer){
           szPActive = this->previousLayer->active.size();
       }
       size_t szTActive = this->active.size();
       for(size_t ta = 0; ta < szPActive; ++ta){
           for(size_t na = 0; na < szTActive; ++na){
               auto & conn = this->drivers.connections[previousLayer->active[ta]][active[na]];
               WeightAdamOptimization::update(conn,epoch, INeuron::learningRate);
           }
       }*/
}
