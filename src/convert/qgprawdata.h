#ifndef QGPRAWDATA_H
#define QGPRAWDATA_H
#include "environment.h"
#include "qgpdimension.h"

#include <module.h>


class QGPRawData
{
public:
    Module * module = nullptr;

    size_t nFilesPerClass = 5;
    size_t eventsPerFile = 1000;

    size_t curFile = 0;
    size_t curEvent = 0;

    std::string inFilepath;
    std::string inputfilename = "phsd50csr.auau.31.2gev.centr.";

    size_t inputzerofill = 5;
    size_t outputzerofill = 6;

    const number maxMomentum = 11;

    std::string suffix = ".dat";

    /**
     * list with particle standard codes
     **/
    std::vector<int> particles = { -211,
                                    111,
                                    211,
                                   2112,
                                   2212,
                                    311,
                                    321,
                                    221,
                                   3122,
                                   -321,
                                   -311,
                                   3212,
                                   3222,
                                   3112,
                                   /*3322,
                                   3312,
                                     22,
                                  -2112,
                                  -2212,
                                  -3122,
                                  -3212,
                                  -3112,
                                  -3222,
                                   3334,
                                  -3322,
                                  -3312,
                                    333,
                                  -3334*/
                                 };

    /**
     * A predefined Momentumscale which can be used to look up the coresponding momentum bin
     *
     * this scale was calculated by
     *          upperBoundary(k) = exp(ln(maxMomentum +1) * k / numberOfBins) - 1
     * where maxMomentum = 11 and numberOfBins = 20
     **/
    std::vector<number> momentumscale = {0.13229363,  0.28208885,  0.45170104,  0.64375183,  0.86120972,  1.1074359,
                                         1.38623623,  1.70192008,  2.05936688,  2.46410162,  2.92238018,  3.44128607,
                                         4.02883991,  4.69412337,  5.44741959,  6.3003721,   7.26616479,  8.3597257,
                                         9.59795775, 11. };

    /**
     * A predefined polarscale which can be applied
     */
    std::vector<number> polarscale = {};

    std::vector<number> inclinationscale = {};


    /**
     * the maximum of all momentum for the provided dataset
     */
    std::vector<number> mom_part_max = {6.75319,6.78448,6.33755,9.44445,10.1197,6.7466,6.71568,7.89404,8.17036,6.27027,6.33039,7.68969,7.25361,7.11498,6.33648,7.8553,2.30947,5.11537,4.70328,4.92913,3.71188,4.96427,4.89566,5.43528,4.94381,3.3811,3.0296,2.73767                                        };

    /**
     * the maximum of all momentum for the provided dataset
     */
    std::vector<number> mom_part_min = {0.00200159,0.00280583,0.00165913,0.00490962,0.00847817,0.00843059,0.00465147,0.0144478,0.00651188,0.0114176,0.00284353,0.0212607,0.0154844,0.0203802,0.0305231,0.0368748,0.0239195,0.0309566,0.0342788,0.0451887,0.0508988,0.0551047,0.0987333,0.118399,0.108897,0.103959,0.107556,0.0740527};




public:
    QGPRawData() = default;


    size_t index(size_t inc, size_t pol, size_t mom, size_t particle){
        return module->fileDimension.index(4,inc,pol,mom,particle);
        /* todo check and delete
        return dim.gridsize[QGPdim::Inclination] * dim.gridsize[QGPdim::Polar]*dim.gridsize[QGPdim::Momentum]*particle
                +dim.gridsize[QGPdim::Inclination] * dim.gridsize[QGPdim::Polar]*mom
                +dim.gridsize[QGPdim::Inclination] * pol
                + inc;*/
    }

    void convert(Module * module);



};

#endif // QGPRAWDATA_H
