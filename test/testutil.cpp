#include "testutil.h"


std::vector<number> applyMask(std::vector<number> vec, std::vector<size_t> mask)
{
    std::vector<number> rvec;
    rvec.resize(mask.size());
    if(mask.size() > 0){
        for(size_t i = 0; i < mask.size(); i++){
            rvec[i] = vec[i];
        }
    }else{
        return vec;
    }
    return rvec;
}

std::vector<number> makeDense(std::vector<number> vec, std::vector<size_t> mask, size_t fullsize){
    std::vector<number> rvec;
    rvec.resize(fullsize);
    if(mask.size() > 0){
        for(size_t i = 0; i < mask.size(); i++){
            rvec[mask[i]] = vec[i];
        }
    } else {
        for(size_t i = 0; i < vec.size(); i++){
            rvec[mask[i]] = vec[i];
        }
    }
    return rvec;
}
