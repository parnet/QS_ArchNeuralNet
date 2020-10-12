#ifndef NEURALNET_H
#define NEURALNET_H

#include <qgpbatch.h>
#include "layerfactory.h"
#include "topology.h"


class NeuralNet
{
public:
    std::vector<AbstractLayer *> layers;
public:
    NeuralNet();

    NeuralNet(Topology topology);

    ~NeuralNet();

    void change(Topology &topology);

    void reset();

    void apply(Topology &topology);

    static NeuralNet *fromFile(std::string &filepath);

    void toFile(const std::string &filepath) const;

    void toM(const std::string & filepath) const;

    /*
     * For Batchsize > 1
     */
    void prepare(QGPBatch * batch, bool training);

    void feedForward(QGPBatch * batch);

    void getResults(std::vector<std::vector<number>> &results);

    void getResults(std::vector<std::vector<number>> &results, std::vector<number> & loss);

    void backprop(QGPBatch * batch, size_t epoch);

    /*
     * For Batchsize == 1
     */
    void prepare(QGPData *data, bool training);

    void feedForward(QGPData *data); // todo change?

    void getResult(std::vector<number> & result);

    void getResult(std::vector<number> & result, double & loss);

    void backprop(QGPData * data, size_t epoch);


};

#endif // NEURALNET_H
