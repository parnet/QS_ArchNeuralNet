#include "qgpdata.h"
#include "fstream"

QGPData::QGPData() {
    this->data.resize(QGPData::size);
    std::fill(&this->data[0], &this->data[QGPData::size], 0);
}



QGPData QGPData::fromFile(std::string filename){
    return QGPData::fromFile(filename, nullptr);
}

QGPData QGPData::fromFile(std::string filename, bool * active){
    QGPData qgp = QGPData();

    std::ifstream file(filename.c_str());
    if(file.fail()){
        qDebug() << "error opening files";
        return qgp;
    }

    if (!file.eof()) {
        TIOStorageDataType input;
        for (size_t i = 0; i != QGPData::size; ++i) {
            file >> input;
            qgp.data[i] = TInternalDataType(input);
            if (active != nullptr && input != 0) { active[i] = true; }
        }
    }
    file.close();
    return qgp;
}

void QGPData::toFile(std::string &filename){
    std::ofstream file((filename.c_str()));
    if(file.fail()){
        qDebug() << "could not create file";
        return;
    }
    for (size_t i = 0; i < QGPData::size; ++i) {
        file << TIOStorageDataType(this->data[i]) << " ";
    }
    file.close();
}


std::vector<number> QGPData::target()
{
    return QGPData::target(this->classification);
}

std::vector<number> QGPData::target(QGPData::TClassificationType classification){
    std::vector<number> vec;
    for (size_t i = 0; i < QGPData::numberOfClassification; i++) {
        vec.push_back(0.0);
    }
    vec[classification] = 1.0;
    return vec;
}
