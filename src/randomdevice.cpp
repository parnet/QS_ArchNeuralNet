#include "randomdevice.h"


long unsigned int RandomDevice::seed = static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
std::default_random_engine RandomDevice::engine = std::default_random_engine(1024/*RandomDevice::seed*/);


std::default_random_engine &RandomDevice::get() {
    return RandomDevice::engine;
}

unsigned int RandomDevice::getSeed() {
    return RandomDevice::seed;
}

void RandomDevice::setSeed(unsigned int seed) {
    RandomDevice::seed = seed;
    RandomDevice::engine = std::default_random_engine(RandomDevice::seed);
}

void RandomDevice::setSeed() {
    RandomDevice::seed = static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    RandomDevice::engine = std::default_random_engine(RandomDevice::seed);
}

std::vector<number> RandomDevice::createUniformVector(size_t size, number from, number to){
    std::vector<number> vec;
    vec.resize(size);
    auto &engine = RandomDevice::engine;
    auto dist = std::uniform_real_distribution<number>(from, to);
    for(size_t i = 0; i < size; ++i){
        vec[i] = dist(engine);
    }
    return vec;
}

std::vector<number> RandomDevice::createNormalVector(size_t size, number mean, number variance){
    std::vector<number> vec;
    vec.resize(size);
    auto &engine = RandomDevice::engine;
    auto dist = std::normal_distribution<number>(mean, variance);
    for(size_t i = 0; i < size; ++i){
        vec[i] = dist(engine);
    }
    return vec;
}

std::vector<size_t> RandomDevice::createMask(size_t size, number prob){
    std::vector<size_t> vec;
    auto &engine = RandomDevice::engine;
    auto dist = std::bernoulli_distribution(prob);
    for(size_t i = 0; i < size; ++i){
        if(dist(engine)){
            vec.push_back(i);
        }
    }
    return vec;
}
