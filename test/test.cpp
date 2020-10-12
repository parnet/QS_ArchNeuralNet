#ifndef TEST_H
#define TEST_H

#include <layer/testfullyconnectedlayer.h>
#include <layer/testmaxpoolinglayer.h>
#include <layer/testconvolutionlayer.h>
#include <layer/testsconvolutionlayer.h>


bool __test(){
    int failed = 0;
    failed += __TestFullyConnectedLayer::all();
    failed += __TestMaxPoolingLayer::all();
    failed += __TestConvolutionLayer::all();
    failed += __TestSConvolutionLayer::all();

    sDebug() << "====================================================================== ";
    sDebug() << "Total fails: " << failed;
    sDebug() << "====================================================================== ";

    if(failed != 0){
        sDebug() << "Not passed";
        exit(240);
    }
}


#endif // TEST_H
