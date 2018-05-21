//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_COL2IM_H
#define LIBND4J_COL2IM_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void _col2im(nd4j::graph::LaunchContext& context, T *dst, T *src, Nd4jLong *outShape, Nd4jLong *inShape, int sY, int sX, int pY, int pX, int imgY, int imgX, int dY, int dX);
        }
    }
}


#endif //LIBND4J_COL2IM_H
