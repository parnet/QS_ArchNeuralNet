#ifndef QGPDIMENSION_H
#define QGPDIMENSION_H

enum QGPdim {
    Inclination = 0, // Azimut [0, 2pi]
    Polar = 1,  // [0, pi]
    Momentum = 2, // [0, \infinity]
    Particle = 3, // type \in {0,1, ... 27}
};

enum QGPSpheric {

};

#endif // QGPDIMENSION_H
