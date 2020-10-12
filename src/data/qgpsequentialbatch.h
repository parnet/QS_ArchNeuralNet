#ifndef QGPSEQUENTIALBATCH_H
#define QGPSEQUENTIALBATCH_H

#include "qgpbatch.h"

#include <module.h>

class QGPSequentialBatch : public QGPBatch
{
public:
    QGPSequentialBatch();

    QGPSequentialBatch(Module * module);

    virtual ~QGPSequentialBatch();

    void load() override;
};

#endif // QGPSEQUENTIALBATCH_H
