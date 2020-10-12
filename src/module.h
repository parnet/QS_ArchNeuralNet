#ifndef MODULE_H
#define MODULE_H

#include "dimension.h"

#include <filename.h>


class Module {
public:
    Module();
    ~Module();

public:
    Filename filename{};
    Dimension fileDimension{};

    void selectPath(std::string & path);

    void selectPath(std::string path);
};

#endif // MODULE_H
