/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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

template <typename T>
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
    static NDArray evalHHmatrix(const NDArray& x);

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
    static void evalHHmatrixData(const NDArray& x, NDArray& tail, T& coeff, T& normX);

    static void evalHHmatrixDataI(const NDArray& x, T& coeff, T& normX);

    /**
    *  this method mathematically multiplies input matrix on Householder from the left P * matrix
    * 
    *  x - input matrix
    *  tail - the essential part of the Householder vector w: [w1, w2, w3, ...]
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */
    static void mulLeft(NDArray& matrix, const NDArray& tail, const T coeff);

    /**
    *  this method mathematically multiplies input matrix on Householder from the right matrix * P
    * 
    *  matrix - input matrix
    *  tail - the essential part of the Householder vector w: [w1, w2, w3, ...]
    *  coeff - scalar, scaling factor in Householder matrix formula  
    */                       
    static void mulRight(NDArray& matrix, const NDArray& tail, const T coeff);
        


};

    
    // /**
    // *  this function reduce given matrix to  upper bidiagonal form (in-place operation), matrix must satisfy following condition rows >= cols
    // * 
    // *  matrix - input 2D matrix to be reduced to upper bidiagonal from    
    // */
    // template <typename T>
    // void biDiagonalizeUp(NDArray& matrix);

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
    // void svd(const NDArray& matrix, NDArray& u, NDArray& s, NDArray& v, const bool calcUV = false, const bool fullUV = false)    



}
}
}


#endif //LIBND4J_HOUSEHOLDER_H
