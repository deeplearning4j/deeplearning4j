//
// Created by Yurii Shyrma on 18.12.2017.
//

#ifndef LIBND4J_HOUSEHOLDER_H
#define LIBND4J_HOUSEHOLDER_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {

    /**
    *  this function evaluates data (coeff, normX, tail) used in Householder transformation
    *  formula for Householder matrix: P = identity_matrix - coeff * w * w^T
    *  P * x = [normX, 0, 0 , 0, ...]
    *  coeff - scalar    
    *  w = [1, w1, w2, w3, ...], "tail" is w except first unity element, that is "tail" = [w1, w2, w3, ...]
    *  w = u / u0
    *  u = x - |x|*e0
    *  u0 = x0 - |x| 
    *  e0 = [1, 0, 0 , 0, ...]
    * 
    *  x - input vector, remains unaffected
    *  tail - output vector with length = x.lengthOf() - 1 and contains all elements of w vector except first one 
    *  normX - this scalar would be the first non-zero element in result of Householder transformation -> P*x  
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                  	
    template <typename T>
    void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff);

    
    

}
}
}


#endif //LIBND4J_HOUSEHOLDER_H
