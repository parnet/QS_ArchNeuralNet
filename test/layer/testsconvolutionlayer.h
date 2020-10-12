#ifndef TESTSCONVOLUTIONLAYER_H
#define TESTSCONVOLUTIONLAYER_H

#include "environment.h"

#include <sconvolutionallayer.h>

class __TestSConvolutionLayer
{
public:
    static bool checkBias(std::vector<number> expected, SConvolutionalLayer * clayer);

    static bool checkKernel(std::vector<std::vector<number>> expected, SConvolutionalLayer * clayer);

public:
    __TestSConvolutionLayer() = delete;

    static int all();

    static bool cycle_1_dd_dd_1();

    static bool cycle_1_dd_sd_1();

    static bool cycle_1_dd_dd_2();

    static bool cycle_1_dd_sd_2();

    static bool cycle_2_dd_dd_1();

    static bool cycle_2_dd_sd_1();

    static bool cycle_2_dd_dd_2();

    static bool cycle_2_dd_sd_2();


    static bool cycle_1_ds_dd_1();

    static bool cycle_1_ds_sd_1();

    static bool cycle_1_ds_dd_2();

    static bool cycle_1_ds_sd_2();

    static bool cycle_2_ds_dd_1();

    static bool cycle_2_ds_sd_1();

    static bool cycle_2_ds_dd_2();

    static bool cycle_2_ds_sd_2();

};

#endif // TESTSCONVOLUTIONLAYER_H
