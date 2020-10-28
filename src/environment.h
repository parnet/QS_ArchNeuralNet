#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

// #include <omp.h>
#include <limits>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <vector>

/* ************************************************************************************************
 * definition to switch from a QT version to a terminal version
 * ********************************************************************************************* */
#ifdef CONSOLE
    #define qDebug() std::cout
    #define sDebug() std::cout
    #define sErr() std::cerr
#define sEndL() std::endl
#else
    #include <QDebug>
    #include <QFile>
    #include <QTextStream>
    #define USEDEBUGWRITER 0 // write everything to a file
    #define sDebug() qDebug()
    #define sEndL() ""
#endif


/* ************************************************************************************************
 * dimensions of the files
 * ********************************************************************************************* */
#define GS_PARTICLES 28
#define GS_MOMENTUM 20
#define GS_INCLINATION 20
#define GS_AZIMUT 20

#define EXPERIMENTAL 1
#define RAND_FIXED 0
#define RAND_NONCXX11 1
#define WRITE_PRECISION 15

typedef double number;
typedef double init_number;




#endif // ENVIRONMENT_H
