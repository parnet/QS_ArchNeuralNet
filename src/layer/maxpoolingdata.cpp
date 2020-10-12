#include "maxpoolingdata.h"
#include "environment.h"
MaxPoolingData::MaxPoolingData(){
    std::vector<number> input;
    std::vector<size_t> trace;
    std::vector<number> output;
    std::vector<number> outputErrorSignal; // right
    std::vector<number> poolingErrorSignal; // left
}

void MaxPoolingData::displayInput(){
    /*
    sDebug() << "=============== Pooling (Activated) ===============";
    for (size_t och = 0; och < this->driver->outChannel; och++) {
        sDebug() << "Output Channel " << och;
        number *cPoolingActivated = &input[och * this->driver->szPooling];
        std::stringstream ss;
        for (size_t i = 0; i < this->driver->szPooling; i++) {
            ss << cPoolingActivated[i] << ", ";

            size_t d = this->driver->dimPooling.dim -1;
            size_t div = this->driver->dimPooling.gridsize[d];
            while((i+1) % div == 0 && d != 0){
                div = div * this->driver->dimPooling.gridsize[d-1];
                sDebug() << ss.str().c_str();
                ss = std::stringstream();
                d--;
            }
        }
        sDebug() << ss.str().c_str() << "\n";
    }*/
}

void MaxPoolingData::displayOutput()
{
   /*sDebug() << "=============== Output Data ===============";
   for (size_t och = 0; och < this->driver->.outChannel; och++) {
       sDebug() << "Output Channel " << och;
       number *cOutput = &output[och * this->driver->.szOutput];
       std::stringstream ss;
       for (size_t i = 0; i < this->driver->.szOutput; i++) {
           ss << cOutput[i] << ", ";

           size_t d = this->driver->.dimOutput.dim -1;
           size_t div = this->driver->.dimOutput.gridsize[d];
           while((i+1) % div == 0 && d != 0){
               div = div * this->driver->.dimOutput.gridsize[d-1];
               sDebug() << ss.str().c_str();
               ss = std::stringstream();
               d--;
           }
       }
       sDebug() << ss.str().c_str() << "\n";
   }*/
}

void MaxPoolingData::displayRightErrorSignal()
{
   /*sDebug() << "============ output error signal ================";
   for (size_t och = 0; och < this->driver->outChannel; och++) {
       number *cOutputErrorSignal = &outputErrorSignal[och * this->driver->szOutput];
       std::stringstream ss;
       for (size_t i = 0; i < this->driver->szOutput; i++) {
           ss << cOutputErrorSignal[i] << ", ";

           size_t d = this->driver->dimOutput.dim -1;
           size_t div = this->driver->dimOutput.gridsize[d];
           while((i+1) % div == 0 && d != 0){
               div = div * this->driver->dimOutput.gridsize[d-1];
               sDebug() << ss.str().c_str();
               ss = std::stringstream();
               d--;
           }
       }
       sDebug() << ss.str().c_str();
   }*/
}

void MaxPoolingData::displayTrace()
{
   /*sDebug() << "============ pooling trace ================";

   for (size_t och = 0; och < this->driver->outChannel; och++) {
       size_t *cTrace = &trace[och * this->driver->szOutput];
       number *cPoolingActivated = &poolingactivated[och * this->driver->szPooling];
       sDebug() << "Output Channel " << och;
       std::stringstream ss;
       for (size_t i = 0; i < this->driver->szOutput; i++) {
           sDebug() << cTrace[i] << " (" << cPoolingActivated[cTrace[i]]<<"), ";

           size_t d = this->driver->dimPooling.dim -1;
           size_t div = this->driver->dimPooling.gridsize[d];
           while((i+1) % div == 0 && d != 0){
               div = div * this->driver->dimPooling.gridsize[d-1];
               sDebug() << ss.str().c_str();
               ss = std::stringstream();
               d--;
           }
       }
       sDebug() << ss.str().c_str();
   }*/
}

void MaxPoolingData::displayLeftErrorSignal()
{

        /*sDebug() << "============ pooling error signal ================";
        for (size_t och = 0; och < this->driver->outChannel; och++) {
            number *cPoolingErrorSignal = &poolingErrorSignal[och * this->driver->szOutput];
            size_t *cTrace = &trace[och * this->driver->szOutput];
            std::stringstream ss;
            for (size_t i = 0; i < this->driver->szOutput; i++) {
                ss << cPoolingErrorSignal[i] <<","; // << "("<< cTrace[i]<<"), ";

                size_t d = this->driver->dimOutput.dim -1;
                size_t div = this->driver->dimOutput.gridsize[d];
                while((i+1) % div == 0 && d != 0){
                    div = div * this->driver->dimOutput.gridsize[d-1];
                    sDebug() << ss.str().c_str();
                    ss = std::stringstream();
                    d--;
                }
            }
            sDebug() << ss.str().c_str();
        }*/

}
