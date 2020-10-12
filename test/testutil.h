#ifndef UTIL_H
#define UTIL_H

#include "environment.h"

std::vector<number> applyMask(std::vector<number> vec, std::vector<size_t> mask);

std::vector<number> makeDense(std::vector<number> vec, std::vector<size_t> mask, size_t fullsize);

#endif // UTIL_H
