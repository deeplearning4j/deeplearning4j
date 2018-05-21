//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_HELPERS_H
#define LIBND4J_HELPERS_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _im2col(nd4j::graph::LaunchContext& context, T *dst, T *src, Nd4jLong *outShape, Nd4jLong *inShape, int kY, int kX, int sY, int sX, int pY, int pX, int dY, int dX, bool isSameMode, T zeroPadVal);
        }
    }
}

#endif //LIBND4J_HELPERS_H
