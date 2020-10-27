#ifndef COORDINATETRANSFORMATION_H
#define COORDINATETRANSFORMATION_H

#include "environment.h"
#include "indexprovider.h"
#include "qgpdimension.h"
#include <module.h>

class CoordinateTransformation
{
public:
    CoordinateTransformation();
};

// todo check
class ZylindricTransformation{

    LogTransformation momentumIdxProvider;
    CenteredLogTransformation heightIdxProvider;
    InclinationTransformation inclinationIdxProvider;
    void apply(number v1, number v2, number v3, size_t &idxHeight, size_t &idxRadius, size_t &idxPolar)
    {
    
        number p_abs_val = sqrt(v2 * v2 + v3 * v3);
        number p_pol = atan2(v3, v2);
        number p_azm = v1;
    
        idxHeight = momentumIdxProvider.toIndex(p_abs_val);
        idxRadius = heightIdxProvider.toIndex(p_azm);
        idxPolar = inclinationIdxProvider.toIndex(p_pol);
    }

};


class SphericalTransformation{
public:
    LogTransformation momentumIdxProvider;
    PolarTransformation polarIdxProvider;
    InclinationTransformation inclinationIdxProvidor;

    SphericalTransformation(Module * module){
        momentumIdxProvider = LogTransformation(module->fileDimension.gridsize[QGPdim::Momentum], 11.0); // 11 is maximal momentum which should be used
        polarIdxProvider = PolarTransformation(module->fileDimension.gridsize[QGPdim::Polar]);
        inclinationIdxProvidor = InclinationTransformation(module->fileDimension.gridsize[QGPdim::Inclination]);
    }

    void apply(number v1, number v2, number v3, size_t &idxMomentum, size_t &idxPolar, size_t &idxInclination){
        number p_abs_val = sqrt(v1 * v1 + v2 * v2 + v3 * v3); // [ 0, \infinity ) or [0, max Momentum]
        number p_inc = atan2(v2, v1); // [-pi, pi]  // todo inclination 0,pi
        number p_pol = acos(v3 / p_abs_val); // [0, pi)    //todo notation wrong   azimut [-pi, pi] , [0, 2pi]

        idxMomentum = momentumIdxProvider.toIndex(p_abs_val);
        idxPolar = polarIdxProvider.toIndex(p_pol );
        idxInclination = inclinationIdxProvidor.toIndex(p_inc);
    }
};



#endif // COORDINATETRANSFORMATION_H
