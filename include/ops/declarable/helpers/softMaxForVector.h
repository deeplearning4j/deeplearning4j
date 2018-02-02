//
// Created by Yurii Shyrma on 31.01.2018
//

#ifndef LIBND4J_SOFTMAXFORVECTOR_H
#define LIBND4J_SOFTMAXFORVECTOR_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {

template <typename T>
void softMaxForVector(const NDArray<T>& input, NDArray<T>& output);

template <typename T>
void logSoftMaxForVector(const NDArray<T>& input, NDArray<T>& output);
    

}
}
}


#endif //LIBND4J_SOFTMAXFORVECTOR_H
