#include "coordinatetransformation.h"
#include "qgprawdata.h"

#include <qgpstatistics.h>
#include <fstream>
#include "qgpsparsestoreddata.h"

void QGPRawData::convert(Module * module){
    //this->dim = module->fileDimension;
    this->module = module;
    ListLookup particleList;
    particleList.list = this->particles;

    SphericalTransformation coordinateSystem = SphericalTransformation(module);

    std::vector<size_t> curEvent;
    curEvent.resize(2);
    std::vector<QGPStatistics> stats;
    stats.resize(2);
    for(unsigned char qgp = 0; qgp < 2; ++qgp){
        stats[qgp].init(module);
        curEvent[qgp] = 0;
    }


    for(unsigned char c = 0; c < 2*this->nFilesPerClass; ++c){
        unsigned char qgp = c % 2;
        unsigned char curFile = c / 2;

        std::stringstream sfilename;
        sfilename  <<inFilepath << module->filename.subdir[qgp] << inputfilename << std::setfill('0') << std::setw(this->inputzerofill) << curFile + 1 << suffix;
        std::ifstream ifile(sfilename.str());
        qDebug() << sfilename.str().c_str();
        if (!ifile){
            qDebug() << "inputfile error";
            return;
        }
        for (size_t curEventOfFile = 0; curEventOfFile < eventsPerFile; ++curEventOfFile) {
            QGPSparseStoredData data;
            data.classification = qgp;

            int nOfParticles, numberOfSubsequentRun, currentParallelEvent, numberOfParticipants, int_tmp;
            number currentImpact,outOfPlane_Psi,weightOfImpact, outOfPlane_Epsilon, eccentricity, float_tmp;
            const size_t max_len = 10;
            number line[max_len];

            ifile >> nOfParticles >> numberOfSubsequentRun >> currentParallelEvent 
                    >> currentImpact >> weightOfImpact >> numberOfParticipants 
                    >> outOfPlane_Psi >> outOfPlane_Epsilon >> eccentricity 
                    >> float_tmp >> float_tmp >> float_tmp
                  >> float_tmp >> float_tmp;
            char buffer[1000];
            const char ch = '\n';
            ifile.getline(buffer, 0, ch);
            //qDebug() << nOfParticles << " particles found";
            for (int iP = 0; iP < nOfParticles; iP++) {
                ifile.getline(buffer, 999, ch);
                std::stringstream sstream;
                sstream << buffer;

                int real_len = max_len;
                for (size_t i = 0; i < real_len; i++) {
                    sstream >> line[i];
                    if (!sstream) {
                        real_len = i;
                        break;
                    }
                }

                // get particle standard code
                int particleID = int(line[0]);
                size_t idxParticle = particleList.toIndex(particleID);

                // todo delete sanity check
                if (idxParticle > module->fileDimension.gridsize[QGPdim::Particle]){
                    qDebug () << "particle" << idxParticle;
                }

                // skip processing if particle is not selected
                if (idxParticle == module->fileDimension.gridsize[QGPdim::Particle]) {
                    qDebug() << "Skip conversion for particle " << particleID;
                    continue;
                }

                if (real_len < 8) {
                    qDebug() << "Error! Array size is not appropriate! real_len = " << real_len;
                    continue;
                }


                size_t idxMomentum, idxPolar, idxInclination;

                coordinateSystem.apply(line[2],line[3],line[4], idxMomentum, idxPolar, idxInclination );

                if(idxParticle >= module->fileDimension.gridsize[QGPdim::Particle]){
                    qDebug() << "Particle Index" << idxParticle;
                }

                if(idxMomentum >= module->fileDimension.gridsize[QGPdim::Momentum]){
                    qDebug() << "Momentum Index" << idxMomentum;
                }

                if(idxPolar >= module->fileDimension.gridsize[QGPdim::Polar]){
                    qDebug() << "Polar Index" << idxPolar;
                }

                if(idxInclination >= module->fileDimension.gridsize[QGPdim::Inclination]){
                    qDebug() << "Inclination Index" << idxInclination;
                }

                size_t idx = index(idxInclination,idxPolar,idxMomentum,idxParticle);
                if(idx >= module->fileDimension.size()){
                    qDebug() << "Index too big" << idx;
                }

                data.data[idx]++;

                stats[qgp].particle[idxParticle]++;
                stats[qgp].momentum[idxMomentum]++;
                stats[qgp].inclination[idxInclination]++;
                stats[qgp].polar[idxPolar]++;

            }

            std::string tmpstring = module->filename.getAbsoluteFilename(curEvent[qgp],qgp);
            data.toFile(tmpstring);
            stats[qgp].add(data);
            curEvent[qgp]++;
        }
    }

    for(unsigned char qgp = 0; qgp < 2; qgp++){
        qDebug() << "QGP: "<< qgp;

        std::stringstream part;
        for(size_t i = 0; i < module->fileDimension.gridsize[QGPdim::Particle];++i){
            part << stats[qgp].particle[i] << "; ";
        }
        qDebug() << part.str().c_str();


        std::stringstream mome;
        for(size_t i = 0; i < module->fileDimension.gridsize[QGPdim::Momentum];++i){
            mome << stats[qgp].momentum[i] << "; ";
        }
        qDebug() << mome.str().c_str();

        std::stringstream incl;
        for(size_t i = 0; i < this->module->fileDimension.gridsize[QGPdim::Inclination];++i){
            incl << stats[qgp].inclination[i] << "; ";
        }
        qDebug() << incl.str().c_str();

        std::stringstream pola;
        for(size_t i = 0; i < module->fileDimension.gridsize[QGPdim::Polar];++i){
            pola << stats[qgp].polar[i] << "; ";
        }
        qDebug() << pola.str().c_str();

        stats[qgp].toFile(qgp);
        }
}
