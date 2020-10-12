#include "qgpsparsestoreddata.h"
#include <fstream>

QGPSparseStoredData::QGPSparseStoredData() {
    this->data.resize(QGPData::size);
    std::fill(&this->data[0], &this->data[QGPData::size], 0);
}



QGPData QGPSparseStoredData::fromFile(std::string filename){
    return QGPSparseStoredData::fromFile(filename, nullptr);
}

QGPData QGPSparseStoredData::fromFile(std::string filename, bool * active){
    QGPData qgp = QGPData();

    std::ifstream file(filename.c_str());
    if(file.fail()){
        qDebug() << "error opening files";
        return qgp;
    }

    if (!file.eof()) {
        size_t numberOfEntries, numOfParticles;
        file >> numberOfEntries;
        file >> numOfParticles;
        size_t index;
        TIOStorageDataType value;
        for (size_t i = 0; i != numberOfEntries; ++i) {
            file >> index;
            file >> value;
            qgp.data[index] = TInternalDataType(value);
            if (active != nullptr) {
                active[index] = true;
            }
        }
    }
    file.close();
    return qgp;
}

void QGPSparseStoredData::toFile(std::string &filename){
    std::ofstream file((filename.c_str()));
    if(file.fail()){
        qDebug() << "could not create file";
        return;
    }
    size_t nonzero = 0;
    size_t numOfParticle = 0;
    for (size_t i = 0; i < QGPData::size; ++i) {
        if(this->data[i] != 0){
            nonzero++;
            numOfParticle += this->data[i];
        }
    }
    file << nonzero << " " << numOfParticle << " ";
    for (size_t i = 0; i < QGPData::size; ++i) {
        if(this->data[i] != 0){
            file << i << " " << TIOStorageDataType(this->data[i]) << " ";
        }
    }
    file.close();
}

void QGPSparseStoredData::toBinaryFile(std::string &filename){
    std::ofstream file((filename.c_str()));
    if(file.fail()){
        qDebug() << "could not create file";
        return;
    }
    size_t nonzero = 0;
    for (size_t i = 0; i < QGPData::size; ++i) {
        if(this->data[i] != 0){
            nonzero++;
        }
    }
    file.write((const char*) &nonzero, sizeof(size_t));
    for (size_t i = 0; i < QGPData::size; ++i) {
        if(this->data[i] != 0){
            file.write((const char*) &i, sizeof(size_t));
            file.write((const char*) (this->data[i]), sizeof(TInternalDataType));
        }
    }
    file.close();
}

std::vector<number> QGPSparseStoredData::target()
{
    return QGPData::target(this->classification);
}

std::vector<number> QGPSparseStoredData::target(QGPSparseStoredData::TClassificationType classification){
    std::vector<number> vec;
    for (size_t i = 0; i < QGPData::numberOfClassification; i++) {
        vec.push_back(0.0);
    }
    vec[classification] = 1.0;
    return vec;
}
