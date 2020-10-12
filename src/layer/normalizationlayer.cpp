#include "normalizationlayer.h"

NormalizationLayer::NormalizationLayer():AbstractLayer(){

}

NormalizationLayer::NormalizationLayer(AbstractLayer *prev): AbstractLayer(prev){

    this->leftActive = prev->rightActive;
    this->rightActive = this->leftActive;

    size_t numberOfNeurons = 2; // todo

    driver.cGlidingMean.resize(numberOfNeurons);
    driver.cGlidingVariance.resize(numberOfNeurons, {1.0});
    driver.cBeta.resize(numberOfNeurons, {0.0});
    driver.cGamma.resize(numberOfNeurons, {1.0});
}

NormalizationLayer::NormalizationLayer(std::istream &stream, AbstractLayer * prev) : AbstractLayer(prev){

}

void NormalizationLayer::init(){

}

void NormalizationLayer::prepare(){
    this->size = previousLayer->size;
    this->data.resize(this->size);
}

void NormalizationLayer::feedforward(){
    std::vector<size_t> activeIndex; // todo
    activeIndex.resize(2);
    for(size_t i = 0; i < 2; i++){
        activeIndex[i] = i;
    }

    const size_t szActive = activeIndex.size();
    const size_t szBatchsize = this->size;

    // calculate mean
    this->aMean.resize(szActive);
    std::fill(&aMean[0],&aMean[szActive],0.0);
    for(size_t bi = 0; bi < szBatchsize; bi++){
        auto &abInput = this->getInput(bi);
        for(size_t i = 0; i < szActive; i++){
            aMean[i] += abInput[i] / number(szBatchsize);
        }
    }


    // calculate variance
    this->aVariance.resize(szActive);
    std::fill(&aVariance[0],&aVariance[szActive],0.0);
        for(size_t bi = 0; bi < szBatchsize; bi++){
        auto &abInput = this->getInput(bi);
        for(size_t i = 0; i < szActive; i++){
            aVariance[i] += pow((abInput[i] - aMean[i]),2) / number(szBatchsize);
        }
    }

    // normalize data
    for(size_t bi = 0; bi < szBatchsize; bi++){
        auto &abInput = this->getInput(bi);
        data[bi].normalized.resize(szActive);

        for(size_t i = 0; i < szActive; i++){
            data[bi].normalized[i] = (abInput[i] - aMean[i]) / sqrt(aVariance[i] + driver.bn_epsilon);
        }
    }

    // calculate output
    for(size_t bi = 0; bi < szBatchsize; bi++){
        auto &tmpnormalized = data[bi].normalized;
        auto &output = data[bi].output;
        output.resize(szActive);
        for(size_t i = 0; i < szActive; i++){
            output[i] = driver.cGamma[activeIndex[i]].weight * tmpnormalized[i] + driver.cBeta[activeIndex[i]].weight;
        }
    }
}

void NormalizationLayer::backprop(){
    std::vector<size_t> activeIndex;
    activeIndex.resize(2);
    for(size_t i = 0; i< 2; i++){
        activeIndex[i] = i;
    }

    const size_t szBatch = this->size;
    const size_t  szActive = activeIndex.size();

     std::vector<std::vector<number>> centered;
     centered.resize(szBatch);


     for(size_t bi = 0; bi < szBatch; bi++){
         centered[bi].resize(szActive);
         auto & input = this->getInput(bi);
         for(size_t i = 0; i < szActive; i++){
            centered[bi][i] = input[i] - aMean[i];
        }
    }
    std::vector<number> precision;
    precision.resize(szActive);
    for(size_t i = 0; i < szActive; i++){
        precision[i] = 1.0 / sqrt(aVariance[i] + driver.bn_epsilon);
    }


    std::vector<std::vector<number>> changeOfNormalized;
    changeOfNormalized.resize(szBatch);

    for(size_t bi = 0; bi < szBatch; bi++){
        auto & tmpRightErrorSignal = this->getRightErrorSignal(bi);
        changeOfNormalized[bi].resize(szActive);
        for(size_t i = 0; i < szActive; i++ ){
            changeOfNormalized[bi][i] = tmpRightErrorSignal[i] * driver.cGamma[activeIndex[i]].weight;
        }
    }

    std::vector<number> changeOfVariance;
    changeOfVariance.resize(szActive);
    std::fill(&changeOfVariance[0], &changeOfVariance[szActive],0.0);
    // changeOfVariance.assign(szActive, 0)
    for(size_t bi = 0; bi < szBatch; bi++){
        for(size_t i = 0; i < szActive; i++){
        changeOfVariance[i] += (-0.5)* pow(precision[i],3) * (changeOfNormalized[bi][i] * centered[bi][i]);
        }
    }

    std::vector<number> changeOfMean;
    changeOfMean.resize(szActive);
    std::fill(&changeOfMean[0],&changeOfMean[szActive],0.0);
    for(size_t bi = 0; bi < szBatch; bi++){
        for(size_t i = 0; i < szActive; i++){
        changeOfMean[i] +=  (-precision[i]) * changeOfNormalized[bi][i] ;
        changeOfMean[i] += (-2.0)*changeOfVariance[i] / number(szBatch) * centered[bi][i];
        }

    }

    for (size_t bi = 0; bi < szBatch; bi++) {
        auto & tmpLeftErrorSignal = this->getLeftErrorSignal(bi);
        tmpLeftErrorSignal.resize(szActive);
        for (size_t i = 0; i < szActive; i++) {
            tmpLeftErrorSignal[i] = (changeOfNormalized[bi][i] * precision[i]) + (changeOfVariance[i] * 2 * centered[bi][i] / number(szBatch)) + (changeOfMean[i] / number(szBatch));
        }
    }

    for(size_t bi = 0;bi < szBatch;bi++){
        auto & tmpRightErrorSignal = this->getRightErrorSignal(bi);
        for(size_t i = 0; i < szActive; i++){
        driver.cGamma[activeIndex[i]].gradient += tmpRightErrorSignal[i] * data[bi].normalized[i];
        }
    }

    // gamma = alpha_gamma * oldgamma + (1-alpha_gamma) * gamma
    for(size_t bi = 0; bi < szBatch; bi++){
        auto & tmpRightErrorSignal = this->getRightErrorSignal(bi);
        for(size_t i = 0; i < szActive; i++){
            driver.cBeta[activeIndex[i]].gradient += tmpRightErrorSignal[i];
        }
    }
}

void NormalizationLayer::update(size_t epoch){
    std::vector<size_t> activeIndex;
    activeIndex.resize(2);
    for(size_t i= 0; i < 2; i++){
        activeIndex[i] = i;
    }

    const size_t szActive = activeIndex.size();
    for(size_t i = 0; i < szActive; i++){
        const size_t indexOfActiveNeuron = activeIndex[i];
        driver.updaterBeta.update(driver.cBeta[indexOfActiveNeuron],epoch); // todo adjust update methods
        driver.updaterGamma.update(driver.cGamma[indexOfActiveNeuron],epoch);
        driver.updaterMean.update(driver.cGlidingMean[indexOfActiveNeuron],epoch);
        driver.updaterVariance.update(driver.cGlidingVariance[indexOfActiveNeuron],epoch);
    }
}

void NormalizationLayer::serialize(std::ostream &out){
    // todo
}

std::vector<number> &NormalizationLayer::getInput(size_t index){
    return previousLayer->getOutput(index);
}

std::vector<number> &NormalizationLayer::getOutput(size_t index){
    return this->data[index].output;
}


std::vector<number> &NormalizationLayer::getRightErrorSignal(size_t index){
    return nextLayer->getLeftErrorSignal(index);
}

std::vector<number> &NormalizationLayer::getLeftErrorSignal(size_t index){
    return this->data[index].leftErrorSignal;
}

