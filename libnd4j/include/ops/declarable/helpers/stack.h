//
// Created by Yurii Shyrma on 02.02.2018
//

#ifndef LIBND4J_STACK_H
#define LIBND4J_STACK_H

#include <ops/declarable/helpers/helpers.h>
#include <NDArray.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

template <typename T>
void stack(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& outArr, const int dim);
    

}
}
}


#endif //LIBND4J_STACK_H
