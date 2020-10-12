#ifndef QGPSPARSEDATA_H
#define QGPSPARSEDATA_H

#include "environment.h"
#include "qgpdata.h"

/**
 * @brief The QGPSparseData class is a sparse implementation of the QGPData where the data only consists out of non zero entries
 */
class QGPSparseData{
public:
    typedef unsigned short TIOStorageDataType;

    typedef unsigned short TInternalDataType;

    typedef unsigned short TClassificationType;

public:
    static constexpr short numberOfClassification = 2;

    static constexpr size_t size = GS_PARTICLES*GS_MOMENTUM*GS_POLAR*GS_INCLINATION;

    static std::vector<number> target(TClassificationType classification);

    static TClassificationType getClassification(std::vector<number> & result);

public:
    TClassificationType classification;

    std::vector<TInternalDataType> data;

    std::vector<size_t> index;
public:
    QGPSparseData();

    static QGPSparseData fromFile(std::string filename, bool *active);

    static QGPSparseData fromFile(std::string filename);

    void toFile(std::string &filename);

    void toBinaryFile(std::string &filename);

    std::vector<number> target();


};
#endif // QGPSPARSEDATA_H
