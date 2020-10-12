#include "testidentityneuron.h"

#include <identityneuron.h>

TestIdentityNeuron::TestIdentityNeuron()
{

}

void TestIdentityNeuron::activate() {
    std::vector<number> {1.0,2.0,-1.0};
    std::vector<number> b = {1.0,2.0,-1.0};
    IdentityNeuron neuron = IdentityNeuron();
    number result = neuron.activate(1, b);
    number expected = 2.0;
    QVERIFY(qFuzzyCompare(result,expected));
}
