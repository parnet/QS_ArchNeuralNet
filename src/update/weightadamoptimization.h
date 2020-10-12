#ifndef WEIGHTADAMOPTIMIZATION_H
#define WEIGHTADAMOPTIMIZATION_H

#include <math.h>
#include "environment.h"

class WeightAdamOptimization {
public:
    struct Variable {
        number weight = 0;
        number vdw = 0;
        number sdw = 0;
        number gradient = 0;
        bool trainable = true;
    };

    WeightAdamOptimization() = default;

    number beta1 = 0.9;
    number beta2 = 0.999;
    number epsilon = 1e-8;

    number learningRate = 0.001;

    void update(Variable & ){
        return;
    }

    void update(Variable & conn, size_t epoch){
        if(conn.trainable){
            if(conn.gradient == 0.0){
                return;
            }
            number new_change_weight = 0;
            number vdwtmp = conn.vdw;
            conn.vdw = beta1 * vdwtmp + (1 - beta1) * conn.gradient;

            number sdwtmp = conn.sdw;
            conn.sdw = beta2 * sdwtmp + (1 - beta2) * pow(conn.gradient, 2.0);

            number vdwCorrected = conn.vdw / (1 - pow(beta1, epoch + 1));
            number sdwCorrected = conn.sdw / (1 - pow(beta2, epoch + 1));

            new_change_weight = vdwCorrected / (sqrt(sdwCorrected) + epsilon);
            conn.weight -= new_change_weight * learningRate;
            conn.gradient = 0; // clear gradient
        }
    }

    number updater(Variable & conn, size_t epoch){
        number re = 0.0;
        if(conn.trainable){
            if(conn.gradient == 0.0){
                return 0;
            }

            number vdwtmp = conn.vdw;
            conn.vdw = beta1 * vdwtmp + (1 - beta1) * conn.gradient;

            number sdwtmp = conn.sdw;
            conn.sdw = beta2 * sdwtmp + (1 - beta2) * pow(conn.gradient, 2.0);

            number vdwCorrected = conn.vdw / (1 - pow(beta1, epoch + 1));
            number sdwCorrected = conn.sdw / (1 - pow(beta2, epoch + 1));

            re = learningRate*vdwCorrected / (sqrt(sdwCorrected) + epsilon);
            conn.weight -= re;
            conn.gradient = 0; // clear gradient
        }
        return re;
    }
};

#endif // WEIGHTADAMOPTIMIZATION_H
