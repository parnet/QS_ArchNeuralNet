#ifndef FILENAME_H
#define FILENAME_H
#include "environment.h"

class Filename {
public:
    Filename() = default;

    std::string path;

    std::string prefix = "phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
    std::string suffix = "_event.dat";

    std::vector<std::string> subdir = {"/qgp/", "/nqgp/"};

    std::string getAbsoluteFilename(size_t index, short classification){
        return path + subdir[classification] + prefix + std::to_string(index) + suffix;
    }
};

#endif // FILENAME_H
