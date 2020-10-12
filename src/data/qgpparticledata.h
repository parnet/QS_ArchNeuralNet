#ifndef QGPPARTICLEDATA_H
#define QGPPARTICLEDATA_H

#include "environment.h"
#include "qgpdata.h"

class QGPParticleData{
public:
    struct Position{
        number momentum;
        number polar;
        number inclination;
    };

    // Particle x (Momentum, Polar, Inclination)
    std::vector<std::vector<Position>> particles;
public:
    QGPParticleData();

    void add(size_t particle, number momentum, number polar, number inclination);

    void toFile(std::string &filename);

    void toBinaryFile(std::string &filename);

    static QGPParticleData fromFile(std::string filename);

    static QGPParticleData fromBinaryFile(std::string filename);

    QGPData toBinData();
};

#endif // QGPPARTICLEDATA_H
