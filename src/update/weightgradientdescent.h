#ifndef WEIGHTGRADIENTDESCENT_H
#define WEIGHTGRADIENTDESCENT_H

#include <stdio.h>
#include "environment.h"

class WeightGradientDescent {
public:
    struct Variable {
        number weight = 0;
        number gradient = 0;
        bool trainable = true;
    };


    number learningRate = 0.01;

    WeightGradientDescent() = default;


    void update(WeightGradientDescent::Variable &conn){
        if(conn.trainable){
            conn.weight -= conn.gradient;
            conn.gradient = 0;
        }
    }


    void update(WeightGradientDescent::Variable & conn, size_t){
        if(conn.trainable){
            conn.weight -= learningRate * conn.gradient;
            conn.gradient = 0;
        }
    }

    number updater(WeightGradientDescent::Variable & conn, size_t, number learningRate){
        number re = 0.0;
        if(conn.trainable){
            re = learningRate * conn.gradient;
            conn.weight -= re;
            conn.gradient = 0;
        }
        return re;
    }
};

#endif // WEIGHTGRADIENTDESCENT_H
