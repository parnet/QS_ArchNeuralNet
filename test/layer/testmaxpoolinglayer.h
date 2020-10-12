#ifndef __TESTMAXPOOLINGLAYER_H
#define __TESTMAXPOOLINGLAYER_H

#include "environment.h"
class __TestMaxPoolingLayer {
public:
    __TestMaxPoolingLayer() = delete;

    static int all();

    static bool cycle_dd_dd();

    static bool cycle_dd_sd();

    static bool forward_sd();

    static bool forward_ss();



};

#endif // __TESTMAXPOOLINGLAYER_H
