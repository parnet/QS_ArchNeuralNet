#ifndef TESTPOOLINXLAYER_H
#define TESTPOOLINXLAYER_H

#include "environment.h"
#include <convolutionallayer.h>

class __TestConvolutionLayer
{
public:
    static bool checkBias(std::vector<number> expected, ConvolutionalLayer * clayer);

    static bool checkKernel(std::vector<std::vector<number>> expected, ConvolutionalLayer * clayer);

public:
    __TestConvolutionLayer() = delete;

    static int all();

    static bool cycle_dd_dd();

    static bool cycle_dd_sd();

    static bool forward_sd();

    static bool forward_ss();

    static bool prepare();
};

#endif // TESTPOOLINXLAYER_H
