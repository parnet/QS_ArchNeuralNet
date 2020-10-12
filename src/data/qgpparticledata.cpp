#include "qgpparticledata.h"


QGPParticleData::QGPParticleData(){


}

void QGPParticleData::add(size_t particle, number momentum, number polar, number inclination){
this->particles[particle].push_back({momentum, polar, inclination});
}

QGPData QGPParticleData::toBinData()
{

}
