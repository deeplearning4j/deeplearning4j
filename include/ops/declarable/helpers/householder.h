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


template<typename T>
class Householder {

    public:
        
    /**
    *  this method calculates Householder matrix P = identity_matrix - coeff * w * w^T
    *  P * x = [normX, 0, 0 , 0, ...]
    *  coeff - scalar    
    *  w = [1, w1, w2, w3, ...]
    *  w = u / u0
    *  u = x - |x|*e0
    *  u0 = x0 - |x| 
    *  e0 = [1, 0, 0 , 0, ...]
    * 
    *  x - input vector, remains unaffected
    */                       
    static NDArray<T> evalHHmatrix(const NDArray<T>& x);

    /**
    *  this method evaluates data required for calculation of Householder matrix P = identity_matrix - coeff * w * w^T
    *  P * x = [normX, 0, 0 , 0, ...]
    *  coeff - scalar    
    *  w = [1, w1, w2, w3, ...]
    *  w = u / u0
    *  u = x - |x|*e0
    *  u0 = x0 - |x| 
    *  e0 = [1, 0, 0 , 0, ...]
    * 
    *  x - input vector, remains unaffected
    *  tail - the essential part of the vector w: [w1, w2, w3, ...]
    *  normX - this scalar is the first non-zero element in vector resulting from Householder transformation -> (P*x)
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                       
    static void evalHHmatrixData(const NDArray<T>& x, NDArray<T>& tail, T& coeff, T& normX);

    static void evalHHmatrixDataI(const NDArray<T>& x, T& coeff, T& normX);

    /**
    *  this method mathematically multiplies input matrix on Householder from the left P * matrix
    * 
    *  x - input matrix
    *  tail - the essential part of the Householder vector w: [w1, w2, w3, ...]
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                       
    static void mulLeft(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff);

    /**
    *  this method mathematically multiplies input matrix on Householder from the right matrix * P
    * 
    *  matrix - input matrix
    *  tail - the essential part of the Householder vector w: [w1, w2, w3, ...]
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                       
    static void mulRight(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff);
        


};

    
    // /**
    // *  this function reduce given matrix to  upper bidiagonal form (in-place operation), matrix must satisfy following condition rows >= cols
    // * 
    // *  matrix - input 2D matrix to be reduced to upper bidiagonal from    
    // */
    // template <typename T>
    // void biDiagonalizeUp(NDArray<T>& matrix);

    // /** 
    // *  given a matrix matrix [m,n], this function computes its singular value decomposition matrix = u * s * v^T
    // *   
    // *  matrix - input 2D matrix to decompose, [m, n]
    // *  u - unitary matrix containing left singular vectors of input matrix, [m, m]
    // *  s - diagonal matrix with singular values of input matrix (non-negative) on the diagonal sorted in decreasing order,
    // *      actually the mathematically correct dimension of s is [m, n], however for memory saving we work with s as vector [1, p], where p is smaller among m and n
    // *  v - unitary matrix containing right singular vectors of input matrix, [n, n]
    // *  calcUV - if true then u and v will be computed, in opposite case function works significantly faster
    // *  fullUV - if false then only p (p is smaller among m and n) first columns of u and v will be calculated and their dimensions in this case are [m, p] and [n, p]
    // *
    // */
    // void svd(const NDArray<T>& matrix, NDArray<T>& u, NDArray<T>& s, NDArray<T>& v, const bool calcUV = false, const bool fullUV = false)    



}
}
}


#endif //LIBND4J_HOUSEHOLDER_H
