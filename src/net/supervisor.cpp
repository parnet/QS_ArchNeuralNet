#include "supervisor.h"
#include <fstream>
#include <qgpsequentialbatch.h>

Supervisor::Supervisor(): neuralNet(
                              #ifdef EXPERIMENTAL
                                                             new NeuralNet(Topology::experimentalTopology())
                              #else
                                                             new NeuralNet(Topology::defaultTopology())
                              #endif
                                                             ) {

}

Supervisor::Supervisor(Module * module): neuralNet(
                              #ifdef EXPERIMENTAL
                                                             new NeuralNet(Topology::experimentalTopology())
                              #else
                                                             new NeuralNet(Topology::defaultTopology())
                              #endif
                                                             ) {
    this->module = module;
}


void Supervisor::training() // todo set module, todo use module for Batch
{
    number summL = 0;
    size_t batchsize = szBatch / 2; // multiplied with number of classes
    size_t trainingIndex = szTraining;
    const size_t batches = trainingIndex / batchsize;

    for (size_t i = 0; i < batches; ++i) {
        QGPBatch * batch = new QGPSequentialBatch(module);
        batch->fromIndex = 0 + batchsize * i;
        batch->toIndex = batchsize * (i+1);
        batch->size = 2 * batchsize;
        batch->load();
        batch->seekActive();

            this->neuralNet->prepare(batch,true);
            this->neuralNet->feedForward(batch);

        std::vector<std::vector<number>> results;
        std::vector<number> loss;
        this->neuralNet->getResults(results,loss);
        size_t bzc = 0;
            for(size_t b = 0 ; b < batch->size; ++b){

                // this->neuralNet->getResults(result);
                //auto target = QGPData::target(batch->data[b].classification);
                summL += loss[b];

                auto detected = QGPSparseData::getClassification(results[b]);
                bool good = (detected == batch->data[b].classification);
                statTotalTraining[curEpoch]++;
                if (good) { statCorrectTraining[curEpoch]++; bzc++; }
                this->curTraining++;

            }
            // qDebug() << bzc << "\n"; // todo endl
            this->neuralNet->backprop(batch, this->curEpoch);

        //qDebug() << "Training " << (statCorrectTraining[curEpoch]) / number(statTotalTraining[curEpoch]) << "\t"
        //         << statCorrectTraining[curEpoch] << "\t" << statTotalTraining[curEpoch];
        delete batch;
    }
    statLossTraining[curEpoch] = summL / statTotalTraining[curEpoch];

}

void Supervisor::validate(){
    size_t valsize = 2; // todo
    size_t numval = szValidation / valsize;

    number summL = 0;
    for(size_t numValSet = 0; numValSet < numval; numValSet++){
        QGPBatch *validationSet = new QGPSequentialBatch(module);

        validationSet->fromIndex = szTraining + numValSet*valsize;
        validationSet->toIndex = szTraining + (numValSet+1)*(valsize);
        validationSet->size = valsize*2;

        validationSet->load();
        validationSet->seekActive();
        this->neuralNet->prepare(validationSet, false);
        this->neuralNet->feedForward(validationSet);

        std::vector<std::vector<number>> results;
        std::vector<number> loss;
        this->neuralNet->getResults(results,loss);
        for(size_t b = 0 ; b < validationSet->size; ++b){

            summL += loss[b];
            auto detected = QGPSparseData::getClassification(results[b]);

            bool good = (detected == validationSet->data[b].classification);
            statTotalValidation[curEpoch]++;
            if (good) { statCorrectValidation[curEpoch]++;}
            curValidation++;
        }
        //qDebug() << statCorrectValidation[curEpoch]  << " of " << statTotalValidation[curEpoch] << ": " << number(statCorrectValidation[curEpoch])/statTotalValidation[curEpoch];
        if(validateonly){curEpoch++;}
        statLossValidation[curEpoch] = summL / this->statTotalValidation[curEpoch];
        delete validationSet;
    }
    if(validateonly){
        size_t valid = 0;
        size_t total = 0;
        for(size_t i = 0; i < this->curEpoch; i++){
            valid += statCorrectValidation[curEpoch];
            total += statTotalValidation[curEpoch];
        }
        sDebug() << double(valid) / total;
    }

    qDebug() << "Validation " << (statCorrectValidation[curEpoch]) / number(statTotalValidation[curEpoch]) << "\t"
             << statCorrectValidation[curEpoch] << "\t" << statTotalValidation[curEpoch];

}

void Supervisor::runTraining()
{
    this->running = true;
    this->curTraining = 0;
    this->curValidation = 0;

    this->clearStats();

    for (this->curEpoch = 0; this->curEpoch < szEpoch; this->curEpoch++) {
        this->curTraining = 0;
        this->curValidation = 0;
        qDebug() << "Epoch " << this->curEpoch;
        training();
        validate();

        qDebug() << "Statistics of epoch:" << endl
                 << "fRunEventsA: " << statTotalTraining[this->curEpoch] << ";   fGoodEventsA: "
                 << statCorrectTraining[this->curEpoch] << endl
                 << "fRunEventsT: " << statTotalValidation[this->curEpoch] << ";   fGoodEventsT: "
                 << statCorrectValidation[this->curEpoch] << endl;
    }
    this->curEpoch = szEpoch;
}

void Supervisor::runValidation()
{
    this->validateonly = true;
    this->running = true;

    bool offset = szTraining % szBatch > 0;
    this->setEpoch(szTraining / szBatch + offset); // todo add missing batch / modulus -> remaining
    this->curTraining = 0;
    this->curValidation = 0;

    this->clearStats();

    this->curEpoch = 0;
    this->curTraining = 0;
    this->curValidation = 0;
    validate();

        qDebug() << "Statistics of epoch:" << endl
                 << "fRunEventsA: " << statTotalTraining[this->curEpoch] << ";   fGoodEventsA: "
                 << statCorrectTraining[this->curEpoch] << endl
                 << "fRunEventsT: " << statTotalValidation[this->curEpoch] << ";   fGoodEventsT: "
                 << statCorrectValidation[this->curEpoch] << endl;

        this->curEpoch = szEpoch;
}

void Supervisor::setBatchSize(size_t number)
{
 this->szBatch = number;
}

void Supervisor::setEpoch(size_t szEpoch){
    this->szEpoch = szEpoch;
    statTotalTraining.resize((szEpoch));
    statCorrectTraining.resize((szEpoch));
    statTotalValidation.resize((szEpoch));
    statCorrectValidation.resize((szEpoch));
    statLossTraining.resize((szEpoch));
    statLossValidation.resize((szEpoch));

    this->clearStats();
}

void Supervisor::setTopology(Topology topology)
{
    this->neuralNet->change(topology);
}

void Supervisor::toFile(const std::string &filepath)
{
    std::ofstream file(filepath.c_str());
        const char * sendl = "\n";
        for(size_t i = 0; i < szEpoch; i++){
            file << number(this->statCorrectTraining[i]) / this->statTotalTraining[i] << ", ";
        }
        file << sendl;
        for(size_t i = 0; i < szEpoch; i++){
            file << number(this->statCorrectValidation[i]) / this->statTotalValidation[i] << ", ";
        }
        file << sendl;

        for(size_t i = 0; i < szEpoch; i++){
            file << this->statLossTraining[i]<< ", ";
        }file << sendl;

        for(size_t i = 0; i < szEpoch; i++){
            file << this->statLossValidation[i]<< ", ";
        } file << sendl;
}

void Supervisor::clearStats()
{
    std::fill(&statTotalTraining[0], &statTotalTraining[szEpoch], 0);
    std::fill(&statCorrectTraining[0], &statCorrectTraining[szEpoch], 0);
    std::fill(&statTotalValidation[0], &statTotalValidation[szEpoch], 0);
    std::fill(&statCorrectValidation[0], &statCorrectValidation[szEpoch], 0);
    std::fill(&statLossTraining[0], &statLossTraining[szEpoch], 0);
    std::fill(&statLossValidation[0], &statLossValidation[szEpoch], 0);
}
