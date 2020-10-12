#ifndef INDEXPROVIDER_H
#define INDEXPROVIDER_H

#include "environment.h"

class IndexProvider
{
public:
    IndexProvider();
};


class ListLookup {
public:
    std::vector<int> list;
    size_t toIndex(int value){
         for (size_t i = 0; i < list.size(); i++) {
                if (list[i] == value){
                  return i;
              }
          }
        return list.size();
    }
};

class ListUpperBoundary {
    std::vector<number> list;
    // todo check list.size > gridsize?

    // todo test method
    size_t toIndex(number value){
    for (size_t i = list.size() -1 ; i > 1 ; i++) {
        if (value < list[i]){
            return i;
        }
    }
    return 0;
    }
};

class ListLowerBoundary {
    std::vector<number> list;
    // todo check list.size > gridsize?

    // todo test method
    size_t toIndex(number value){
    for (size_t i = 0 ; i < list.size() ; i++) {
        if (value > list[i]){
            return i;
        }
    }
    return 0;
    }
};


class LogTransformation {
public:
    size_t gridsize;
    number max;

    LogTransformation(){

    }

    LogTransformation(size_t gridsize, number max){
        this->gridsize = gridsize;
        this->max = max;
    }

    size_t toIndex(number val){
        auto bin= floor(gridsize*log(val+1)/log(max +1));

        if(bin < 0){
            sDebug() << "Warning value is negative for log " << val;
        }

        if(bin >= gridsize){
            sDebug() << "Warning value log is to big" << val;
            bin = gridsize -1;
        }
        return size_t(bin);
    }
};

class CenteredLogTransformation {
public:
    size_t gridsize;
    number max;

    size_t toIndex(number n){
        bool sign = n > 0.0;
        number abs = fabs(n);
        int uneven = gridsize % 2;
        int bin = floor((gridsize/2+uneven) * log(abs+1)/log(max +1));
        size_t sBin;
        if(sign){
            // todo
        } else {

        }

        return sBin;
    }
};

class LinearTransformation {
public:
    size_t gridsize;

    number min;
    number max;

    bool warn = true;

    void setRanges(number min, number max){
        this->min = min;
        this->max = max;
    }

    size_t toIndex(number val){
        auto bin = (val - min)/(max + min)* gridsize;
        if(bin < 0){
            sDebug() << "Warning value is negative for linear " << val;
        }

        if(bin >= gridsize){
            sDebug() << "Warning value linear is to big" << val;
            bin = gridsize -1;
        }

        return size_t(bin);
    }
};


class PolarTransformation {
public:
    size_t gridsize;

    PolarTransformation()= default;

    PolarTransformation(size_t gridsize){
        this->gridsize = gridsize;
    }

    size_t toIndex(number polar){
        auto bin = polar *M_1_PI*this->gridsize;

        if(bin < 0){
            sDebug() << "Warning value is negative for polar " << polar;
        }

        if(bin >= this->gridsize){
            sDebug() << "Warning value polar is too big" << polar;
            bin = this->gridsize -1;
        }

        return size_t(bin);
        }

};


class InclinationTransformation {
public:
    size_t gridsize;
    InclinationTransformation() = default;
    InclinationTransformation(size_t gridsize){
        this->gridsize = gridsize;
    }
    size_t toIndex(number inclination){
        auto bin = ((inclination*M_1_PI + 1.0) /2.0 * gridsize);
        if(bin < 0){
            sDebug() << "Warning value is negative for inclination " << inclination;
        }
        if(bin >= gridsize){
            sDebug() << "Warning value inclination is to big" << inclination;
            bin = gridsize -1;
        }
        return size_t(bin);
    }
};

#endif // INDEXPROVIDER_H
