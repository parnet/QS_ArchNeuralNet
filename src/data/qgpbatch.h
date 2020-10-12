#ifndef QGPBATCH_H
#define QGPBATCH_H

#include "environment.h"
#include "qgpdata.h"
#include "qgpsparsedata.h"

#include <module.h>

class QGPBatch{
public:
    //typedef QGPData FData;
    typedef QGPSparseData LData;
public:
    QGPBatch();

    QGPBatch(Module * module);

    virtual ~QGPBatch();

    virtual void load() = 0;

    void seekActive();


    void begin();

    bool end();

    LData & next();

public:
     std::vector<LData> data;

     bool * nonzero = nullptr;

     size_t position = 0;

     size_t size;

     size_t fromIndex;

     size_t toIndex;

     Module * module;


};

#endif // QGPBATCH_H
