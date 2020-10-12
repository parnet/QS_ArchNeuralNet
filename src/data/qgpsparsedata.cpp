#include "qgpsparsedata.h"
#include "fstream"


QGPSparseData::QGPSparseData() {
}



QGPSparseData QGPSparseData::fromFile(std::string filename){
    return QGPSparseData::fromFile(filename, nullptr);
}

QGPSparseData QGPSparseData::fromFile(std::string filename, bool * active){
    QGPSparseData qgp = QGPSparseData();

    std::ifstream file(filename.c_str());
    if(file.fail()){
        qDebug() << "error opening files";
        return qgp;
    }

    if (!file.eof()) {
        size_t numberOfEntries, numOfParticles;
        file >> numberOfEntries;
        file >> numOfParticles;
        qgp.data.resize(numberOfEntries);
        qgp.index.resize(numberOfEntries);

        size_t index;
        TIOStorageDataType value;
        for (size_t i = 0; i != numberOfEntries; ++i) {
            file >> index;
            file >> value;
            qgp.index[i] = index;
            qgp.data[i] = TInternalDataType(value);
            if (active != nullptr) {
                active[index] = true;
            }
        }
    }
    file.close();
    return qgp;
}

void QGPSparseData::toFile(std::string &filename){
    std::ofstream file((filename.c_str()));
    if(file.fail()){
        qDebug() << "could not create file";
        return;
    }
    size_t nonzero = 0;
    size_t numOfParticle = 0;
    for (size_t i = 0; i < this->data.size(); ++i) {
        if(this->data[i] != 0){
            nonzero++;
            numOfParticle += this->data[i];
        }
    }
    file << nonzero << " " << numOfParticle << " ";
    for (size_t i = 0; i < this->data.size(); ++i) {
        if(this->data[i] != 0){
            file << i << " " << TIOStorageDataType(this->data[i]) << " ";
        }
    }
    file.close();
}

void QGPSparseData::toBinaryFile(std::string &filename){
    std::ofstream file((filename.c_str()));
    if(file.fail()){
        qDebug() << "could not create file";
        return;
    }
    size_t nonzero = 0;
    for (size_t i = 0; i < this->data.size(); ++i) {
        if(this->data[i] != 0){
            nonzero++;
        }
    }
    file.write((const char*) &nonzero, sizeof(size_t));
    for (size_t i = 0; i < this->data.size(); ++i) {
        if(this->data[i] != 0){
            file.write((const char*) &i, sizeof(size_t));
            file.write((const char*) (this->data[i]), sizeof(TInternalDataType));
        }
    }
    file.close();
}

std::vector<number> QGPSparseData::target()
{
    return QGPData::target(this->classification);
}

std::vector<number> QGPSparseData::target(QGPSparseData::TClassificationType classification){
    std::vector<number> vec;
    for (size_t i = 0; i < QGPData::numberOfClassification; i++) {
        vec.push_back(0.0);
    }
    vec[classification] = 1.0;
    return vec;
}

QGPSparseData::TClassificationType QGPSparseData::getClassification(std::vector<number> &result)
{
    number max_value = result[0];
    unsigned char max_index = 0;
    for (unsigned char i = 1; i < QGPSparseData::numberOfClassification; ++i) {
        if (result[i] > max_value) {
            max_value = result[i];
            max_index = i;
        }
    }
    return max_index;
}
