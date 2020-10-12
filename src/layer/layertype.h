 #ifndef LAYERTYPE_H
#define LAYERTYPE_H

enum LayerType {
    Abstract,
    // General
    Activation,
    Input,
    Output,
    // Connections
    FullyConnected,
    SparseConnected,
    PartialConnected,
    // Advanced
    Normalization,
    Dropout, // todo delete?
    // Convolution
    Convolution,
    SelectiveConvolution,
    // Pooling / Downsampling
    MaxPooling,
    AveragePooling,
    EuklidianPooling,
    LpPooling,
    StochasticPooling,
    // Composition
    Composed,
    SubNet,
    Split,
    Agglomeration,
};

#endif // LAYERTYPE_H
