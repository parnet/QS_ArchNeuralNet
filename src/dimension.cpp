#include "dimension.h"

/*
Dimension::Dimension(std::istream &file){
    size_t tmp;
    file >> tmp;
    this->dim = tmp;
    gridsize.resize(dim);
    for(size_t i = 0; i < dim; i++){
        file >> tmp;
        gridsize[i] = tmp;
    }
}

void Dimension::serialize(std::ostream &out){
        const char * sendl = "\n";
        out << dim << " ";
        gridsize.resize(dim);
        for(size_t i = 0; i < dim; i++){
            out << gridsize[i] << " ";
        }
        out << sendl;
}

Dimension::Dimension(size_t sz, ...) : dim(sz){
    va_list var;
    va_start(var,sz);
    gridsize.resize(sz);
    for(size_t i = 0; i < sz; ++i){
        gridsize[i] = va_arg(var,size_t);
    }
}
*/

size_t Dimension::size() const {
    size_t result = gridsize[0];
    for(size_t i = 1; i < dim; i++){
        result *= gridsize[i];
    }
    return result;
}

size_t Dimension::index(size_t sz,...){
    //assert(sz == dim);
    std::va_list var;
    va_start(var, sz);
    size_t index =  *(size_t *)(var + (dim-1)*sizeof(size_t));
    for(size_t d = dim -2; d != 0; --d){
        index = index*gridsize[d] + (*(size_t *)(var + (d)*sizeof(size_t)));
    }
    index = index*gridsize[0] + (*(size_t *)(var));
    return index;
}

size_t Dimension::index(size_t *coords) {
        size_t index = coords[dim-1];
        for(size_t d = dim-2; d != 0; --d){
            index = index*gridsize[(d)] + coords[d];
        }
        index = index * gridsize[0] + coords[0];
        return index;
}
/*
void Dimension::inccoord(size_t *coords)
{
        size_t d = 0;
        while(d<dim-1 && coords[d] == gridsize[d]-1){
            coords[d] = 0;
            d++;
            }
        coords[d]++;
    }






void Dimension::inccoord(size_t *coords, size_t *stride)
{
        size_t d = 0;
        while(d<dim-1 && coords[d] >= gridsize[d]-stride[d]){
            coords[d] = 0;
            d++;
            }
        coords[d] += stride[d] ;
    }
*/

void Dimension::inc(size_t *coords, size_t *lower, size_t *upper){
        size_t d = 0;
        while(d != dim-1 && coords[d] == upper[d]-1){
            coords[d] = lower[d];
            d++;
            }
        coords[d]++;
}

void Dimension::inc(size_t *coords, size_t *stride){
        size_t d = 0;
            for(;d!=dim-1 && coords[d] >= gridsize[d]-stride[d];d++){
                coords[d] = 0;
                }

            coords[d] += stride[d] ;

}
void Dimension::inc(size_t *coords){
        size_t d = 0;
            while(d != dim-1 && coords[d] == gridsize[d]-1){
                coords[d] = 0;
                d++;
                }
     coords[d]++;
    }
/*
size_t Dimension::indexLeast(size_t *coords)
{
        size_t index = coords[dim-1];
        for(size_t d = dim-2; d != 0; --d){
               index = index*gridsize[d+1] + coords[d];
        }
        index = index*gridsize[1] + coords[0];
        return index;
    }
*/
std::vector<size_t> Dimension::zeroCoords(){
    // todo lower / upper boundary
    std::vector<size_t> ret;
    ret.resize(this->dim);
    std::fill(&ret[0],&ret[this->dim],0);
    return ret;
}

/*

std::vector<number> Dimension::zeros(){
        std::vector<number> ret;
        ret.resize(dim);
        return ret;
    }

size_t Dimension::index(size_t sz, ...)
{
        assert(sz == dim);
        std::va_list var;
        va_start(var, sz);
        size_t index = va_arg(var, size_t);
        for(size_t d = 1; d < dim; ++d){
            index = index*gridsize[d] + va_arg(var,size_t);
        }
        return index;
    }

*/
