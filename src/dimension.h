#ifndef DIMENSION_H
#define DIMENSION_H

#include "environment.h"
#include <cstdarg>

class Dimension {
public:
    size_t dim = 4;
    std::vector<size_t> gridsize = {GS_INCLINATION,GS_POLAR,GS_MOMENTUM,GS_PARTICLES};

    Dimension() = default;

    Dimension(std::istream & file);


    void serialize(std::ostream &out);

    Dimension(size_t sz, ...);


    Dimension(const std::vector<size_t> & gridsize) : dim(gridsize.size()){
        this->gridsize = gridsize;
    }

    ~Dimension(){}

    size_t size() const;

    //size_t index(size_t sz, ...);


    size_t index(size_t sz, ...);


    size_t index(size_t * coords);


    void inc(size_t * coords);

    void inccoordLeast(size_t * coords);

    void inc(size_t * coords, size_t * lower, size_t * upper);

    //size_t inc(size_t * coords);

     void inccoord(size_t * coords, size_t * stride);


    void inc(size_t * coords, size_t * stride);

    std::vector<size_t> zeroCoords();

    std::vector<number> zeros();

};

#endif // DIMENSION_H
