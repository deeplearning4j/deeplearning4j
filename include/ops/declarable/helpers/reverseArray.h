//
// Created by Yurii Shyrma on 25.01.2018
//

#ifndef LIBND4J_REVERSEARRAY_H
#define LIBND4J_REVERSEARRAY_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


template <typename T>
void reverseArray(T* inArr, int *inShapeBuffer, T *result, int *zShapeBuffer, int numOfElemsToReverse = 0);


}
}
}

#endif //LIBND4J_REVERSEARRAY_H
