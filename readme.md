##Known issues
while training with the full dataset 4000 for training and 1000 for validation the ram consume in phase of validation is
tremendous (1.7 or 2.7 GB) this is because the validation is performed as one entire batch, this can be easily fixed see
training

batchnormalization: the gliding means and the gliding variances are calculated but currently not used. Instead the
values are calculated from the the validation batch which is not consistent with the papers. this should be changed for
a reliable training and for anlyzing data

dropout and batchnormalization does not work at the same time for the same layer and is not intended to work at all

batchsizes are currently restricted to a size which is a multiple of 2 because qgp and nonqgp files are always loaded at
once. layer.h is intended for single backpropagation and for a single analyzing mode

the statistics are currently not used. performance was more important. to choose the draw statistics mode could result
in undefined behaviour. the statistics should be rather once evaluated while the original .dat files are converted

the seed for rand()  (srand) and for the RandomDevice (::seed) is fixed by the value of 1024 so that debugging is easier
and the different topology types can be compared with the same initial weights and the same dropout behaviour

the fixed dropout of the topology will be overwritten by a decaying dropout

pause is currently not implemented

##Changes
fixed the "create_input.cpp" to fill every marginalized momentum bin (espacially with index 0)
improved the ui to prevent crashes