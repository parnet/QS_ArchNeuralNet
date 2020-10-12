#ifndef QGPDATA_H
#define QGPDATA_H

#include "environment.h"

class QGPData{
    typedef unsigned short TIOStorageDataType;
    typedef unsigned char TInternalDataType;
    typedef unsigned char TClassificationType;
public:
    QGPData();

public:
    static constexpr short numberOfClassification = 2;

    static constexpr size_t size = GS_PARTICLES*GS_MOMENTUM*GS_POLAR*GS_INCLINATION;

    TClassificationType classification;

    std::vector<TInternalDataType> data;

    static QGPData fromFile(std::string filename, bool *active);

    static QGPData fromFile(std::string filename);

    void toFile(std::string &filename);

    std::vector<number> target();

    static std::vector<number> target(TClassificationType classification);
};

#endif // QGPDATA_H
