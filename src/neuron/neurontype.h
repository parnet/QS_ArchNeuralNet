#ifndef NEURONTYPES_H
#define NEURONTYPES_H


enum NeuronType {
    // General
    Identity,
    Bias,
    // Sigmoid
    Logistic,
    Tanh,
    Softmax,
    SoftmaxDiagonal,
    SoftmaxAGI,
    // ReLU
    ReLU,
    LeakyReLU,
};

#endif // NEURONTYPES_H
