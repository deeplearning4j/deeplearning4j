//
// Created by Yurii Shyrma on 07.12.2017.
//

#ifndef LIBND4J_MATRIXSETDIAG_H
#define LIBND4J_MATRIXSETDIAG_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void matrixSetDiag(const NDArray<T>* input, const NDArray<T>* diagonal, NDArray<T>* output);
    

}
}
}


#endif //LIBND4J_MATRIXSETDIAG_H
