#ifndef TESTIDENTITYNEURON_H
#define TESTIDENTITYNEURON_H

#include <QtTest/QtTest>

class TestIdentityNeuron : public QObject
{
public:
    TestIdentityNeuron();
private slots:
    void activate();
};


QTEST_MAIN(TestIdentityNeuron)


#endif // TESTIDENTITYNEURON_H
