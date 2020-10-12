#include "softmaxagineuron.h"

SoftmaxAGINeuron::SoftmaxAGINeuron()
{

}

SoftmaxAGINeuron::~SoftmaxAGINeuron()
{

}

number SoftmaxAGINeuron::activate(size_t index, std::vector<number> &input)
{
    size_t szActive = input.size();
    number min_v = std::numeric_limits<number>::max();
    for (size_t a = 0; a < szActive; ++a) {
        if (input[a] < min_v) {
            min_v = input[a];
        }
    }

    number sum = 0;
    for (size_t a = 0; a < szActive; ++a) {
        number sumval = input[a] - min_v;
        input[a] = sumval;
        sum += exp(sumval);
    }


    return exp(input[index]) / sum;
}

number SoftmaxAGINeuron::backpass(size_t index, std::vector<number> &errorsignal, std::vector<number> &input){
    sDebug() << "Softmax AGI backpass not implemented";
    return 0.0;
}

number SoftmaxAGINeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input)
{
    sDebug() << "Softmax AGI backpass not implemented";
    return 0.0;
}

number SoftmaxAGINeuron::backpass(size_t errorindex, std::vector<number> &errorsignal, size_t inputindex, std::vector<number> &input, size_t outputindex, std::vector<number> &output)
{
    number sum = 0;
    size_t szActive = output.size();
    for (size_t a = 0; a < szActive; ++a) {
      sum += exp(output[a]);
    }

    number difference = errorsignal[errorindex];
    number activation_functionSoftMax = exp(output[outputindex])/sum;
    return difference * activation_functionSoftMax * (1 - activation_functionSoftMax);

}
