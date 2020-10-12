#include "qgpsparsestoreddata.h"
#include "qgpstatistics.h"
#include <fstream>
QGPStatistics::QGPStatistics()
{

}

void QGPStatistics::init(Module *module)
{
    this->module = module;
    particle.resize(module->fileDimension.gridsize[QGPdim::Particle],0);
    inclination.resize(module->fileDimension.gridsize[QGPdim::Inclination],0);
    polar.resize(module->fileDimension.gridsize[QGPdim::Polar],0);
    momentum.resize(module->fileDimension.gridsize[QGPdim::Momentum],0);
    std::fill(&particle[0],&particle[module->fileDimension.gridsize[QGPdim::Particle]],0);
    std::fill(&inclination[0],&inclination[module->fileDimension.gridsize[QGPdim::Inclination]],0);
    std::fill(&polar[0],&polar[module->fileDimension.gridsize[QGPdim::Polar]],0);
    std::fill(&momentum[0],&momentum[module->fileDimension.gridsize[QGPdim::Momentum]],0);

    this->data.resize(module->fileDimension.size(),0);

}

void QGPStatistics::add(QGPData &data){
    for(size_t i = 0; i < this->data.size(); ++i){
        this->data[i] += data.data[i];
    }
}

void QGPStatistics::add(QGPSparseStoredData &data){
    for(size_t i = 0; i < this->data.size(); ++i){
        this->data[i] += data.data[i];
    }
}

void QGPStatistics::toFile(unsigned char qgp)
{
        std::stringstream ss;
        if(qgp == 0 || qgp == 1){
        ss << this->module->filename.path << this->module->filename.subdir[qgp] << "overall.dat";
        } else {
            ss << this->module->filename.path << "overall.dat";

        }

        std::ofstream file(ss.str().c_str());
        if(file.fail()){
            qDebug() << "could not create file";
            return;
        }
        for (size_t i = 0; i < this->data.size(); ++i) {
            file << this->data[i] << " ";
        }
        file.close();

}
