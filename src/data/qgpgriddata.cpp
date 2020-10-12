#include "qgpgriddata.h"
#include "environment.h"

QGPGridData::QGPGridData(){
    struct Particle {
        size_t particleType;
        number momentum;
        number polar;
        number inclination;
    };

    /**
     * @brief the grid with size binMomentum*binPolar*binInclination
     * contains the precise information to a particle.
     */
    std::vector<std::vector<Particle>> grid;

}
