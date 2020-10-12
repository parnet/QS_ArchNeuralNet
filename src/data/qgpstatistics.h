#ifndef QGPSTATISTICS_H
#define QGPSTATISTICS_H

#include "environment.h"
#include "module.h"
#include "qgpdata.h"
#include "qgpdimension.h"
#include "qgpsparsestoreddata.h"

class QGPStatistics
{
public:
    QGPStatistics();
    Module *module;
    void init(Module * module);

    std::vector<size_t> momentum;
    std::vector<size_t> polar;
    std::vector<size_t> inclination;
    std::vector<size_t> particle;

    std::vector<number> data;

    void add(QGPData & data);

    void add(QGPSparseStoredData & data);

    void toFile(unsigned char qgp);

};

#endif // QGPSTATISTICS_H
