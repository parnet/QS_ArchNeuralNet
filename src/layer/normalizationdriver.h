#ifndef NORMALIZATIONDRIVER_H
#define NORMALIZATIONDRIVER_H

#include <weightmomentumofinertia.h>



class NormalizationDriver
{
public:

    typedef WeightMomentumOfInertia MomentumUpdater;

public:
    MomentumUpdater updaterMean;
    MomentumUpdater updaterVariance;
    MomentumUpdater updaterGamma;
    MomentumUpdater updaterBeta;

public:
    number bn_epsilon = 1e-8;

    std::vector<MomentumUpdater::Variable> cGlidingMean;
    std::vector<MomentumUpdater::Variable> cGlidingVariance;

    std::vector<MomentumUpdater::Variable> cGamma; // scaling value
    std::vector<MomentumUpdater::Variable> cBeta; // shift value

public:
    NormalizationDriver();

    void update(size_t epoch);

};

#endif // NORMALIZATIONDRIVER_H
