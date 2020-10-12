#ifndef NEURONFACTORY_H
#define NEURONFACTORY_H

#include "neurontype.h"
#include "biasneuron.h"
#include "identityneuron.h"
#include "leakyreluneuron.h"
#include "logisticneuron.h"
#include "reluneuron.h"
#include "softmaxagineuron.h"
#include "softmaxdiagonalneuron.h"
#include "softmaxneuron.h"
#include "tanhneuron.h"
class NeuronFactory
{
public:
    static AbstractNeuron *create(NeuronType af) {
        if (af == Identity) {
            return new IdentityNeuron();
        } else if (af == LeakyReLU) {
            return new LeakyReLUNeuron();
        } else if(af == ReLU){
            return new ReLUNeuron();
        }else if (af == Logistic) {
            return new LogisticNeuron();
        } else if (af == Softmax) {
            return new SoftmaxNeuron();
        } else if (af == SoftmaxAGI) {
            return new SoftmaxAGINeuron();
        } else if (af == SoftmaxDiagonal) {
          return new SoftmaxDiagonalNeuron();
         } else if(af == Bias){
            return new BiasNeuron();
        }
        sDebug() << "unknown neuron type";
        return nullptr;
    }
};


#endif // NEURONFACTORY_H
