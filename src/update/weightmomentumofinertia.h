#ifndef WEIGHTMOMENTUMOFINERTIA_H
#define WEIGHTMOMENTUMOFINERTIA_H

#include <stdio.h>
#include "environment.h"

class WeightMomentumOfInertia {
public:
    struct Variable {
        number weight = 0;
        number gradient = 0;
        number oldGradient = 0;
        bool trainable = true;
    };



    WeightMomentumOfInertia() = default;

    number alpha = 0.1;
    number learningRate = -1.0;//0.001;



    void update(WeightMomentumOfInertia::Variable &conn){
        if(conn.trainable){
            conn.oldGradient =  (1-alpha)*conn.oldGradient + alpha * conn.gradient;
            conn.weight -= conn.oldGradient;
            conn.gradient = 0;
        }
    }

    void update(WeightMomentumOfInertia::Variable &conn, size_t){
        if(conn.trainable){
            conn.oldGradient =  (1-alpha)*conn.oldGradient + alpha * conn.gradient;
            conn.weight -= learningRate*conn.oldGradient;
            conn.gradient = 0;
        }

    }

    number updater(WeightMomentumOfInertia::Variable &conn, size_t){
        number re = 0.0;
        if(conn.trainable) {
            conn.oldGradient =  (1-alpha)*conn.oldGradient + alpha * conn.gradient;
            re =  learningRate*conn.oldGradient;
            conn.weight -=re;
            conn.gradient = 0;

        }
        return re;
    }

};

#endif // WEIGHTMOMENTUMOFINERTIA_H

