#ifndef RANDOMDEVICE_H
#define RANDOMDEVICE_H
#include <random>
#include <chrono>

#include "environment.h"


class RandomDevice {
private:
    static long unsigned int seed;

public:
    static std::default_random_engine engine;

    RandomDevice() = delete;

    static std::default_random_engine &get();

    static unsigned int getSeed();

    static void setSeed(unsigned int seed);

    static void setSeed();

    static std::vector<number> createUniformVector(size_t size, number from, number to);

    static std::vector<number> createNormalVector(size_t size, number mean, number variance);

    static std::vector<size_t> createMask(size_t size, number prob);
};
#endif
