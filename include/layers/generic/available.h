//
// @author raver119@gmail.com
//

#ifndef PROJECT_AVAILABLE_H
#define PROJECT_AVAILABLE_H

#define NATIVE_LAYERS \
        (0, nd4j::layers::DenseLayer)
//        (1, nd4j::layers::ConvolutionLayer) ,\
//        (2, nd4j::layers::Pooling2DLayer) ,\
//        (3, nd4j::layers::LSTMLayer)

// here we should build list of includes for this backend AUTOMATICALLY, based on list and naming convention
#include "./DenseLayer.h"
#include "./ConvolutionLayer.h"
#include "./Pooling2DLayer.h"
#include "./LSTMLayer.h"

#endif //PROJECT_AVAILABLE_H
