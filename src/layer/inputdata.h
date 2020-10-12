#ifndef INPUTDATA_H
#define INPUTDATA_H
#include "environment.h"

#include <qgpdata.h>
#include <qgpsparsedata.h>
#include <qgpsparsestoreddata.h>

class InputData{

public:
    std::vector<size_t> active;
    std::vector<number> output;
public:
    InputData();

    void setData(QGPSparseData &data);

    void setData(QGPData &data);

    void setData(QGPSparseStoredData &data);


};

#endif // INPUTDATA_H
