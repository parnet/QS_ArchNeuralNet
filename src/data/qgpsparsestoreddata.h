#ifndef QGPSPARSESTOREDDATA_H
#define QGPSPARSESTOREDDATA_H

#include "environment.h"
#include "qgpdata.h"

class QGPSparseStoredData{
public:
    typedef unsigned short TIOStorageDataType;
    typedef unsigned char TInternalDataType;
    typedef unsigned char TClassificationType;
public:
    QGPSparseStoredData();

public:
    static constexpr short numberOfClassification = 2;

    static constexpr size_t size = GS_PARTICLES*GS_MOMENTUM*GS_INCLINATION*GS_AZIMUT;

    TClassificationType classification;

    std::vector<TInternalDataType> data;

    static QGPData fromFile(std::string filename, bool *active);
    static QGPData fromFile(std::string filename);

    void toFile(std::string &filename);
    void toBinaryFile(std::string &filename);

    std::vector<number> target();
    static std::vector<number> target(TClassificationType classification);
};

#endif // QGPSPARSESTOREDDATA_H
