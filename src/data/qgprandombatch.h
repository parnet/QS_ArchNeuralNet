#ifndef QGPRANDOMBATCH_H
#define QGPRANDOMBATCH_H

#include "qgpbatch.h"


class QGPRandomBatch : public QGPBatch
{
public:
    QGPRandomBatch();

    QGPRandomBatch(Module * module);

    ~QGPRandomBatch() override;

    void load() override;
};

#endif // QGPRANDOMBATCH_H
