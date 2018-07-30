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
// Created by Yurii Shyrma on 02.01.2018
//

#ifndef LIBND4J_HHSEQUENCE_H
#define LIBND4J_HHSEQUENCE_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class HHsequence {

    public:
    
    /*
    *  matrix containing the Householder vectors
    */
    NDArray<T> _vectors;        

    /*
    *  vector containing the Householder coefficients
    */
    NDArray<T> _coeffs;    
    
    /*
    *  shift of the Householder sequence 
    */
    int _shift;

    /*
    *  length of the Householder sequence
    */
    int _diagSize;        

    /* 
    *  type of sequence, type = 'u' (acting on columns, left) or type = 'v' (acting on rows, right)
    */
    char _type;        

    /*
    *  constructor
    */
    HHsequence(const NDArray<T>& vectors, const NDArray<T>& coeffs, const char type);

    /**
    *  this method mathematically multiplies input matrix on Householder sequence from the left H0*H1*...Hn * matrix
    * 
    *  matrix - input matrix to be multiplied
    */                       
    void mulLeft(NDArray<T>& matrix) const;

    NDArray<T> getTail(const int idx) const;

    void applyTo(NDArray<T>& dest) const;

    FORCEINLINE int rows() const;

};


//////////////////////////////////////////////////////////////////////////
template<typename T>
FORCEINLINE int HHsequence<T>::rows() const {

    return _type == 'u' ? _vectors.sizeAt(0) : _vectors.sizeAt(1); 
}    



}
}
}


#endif //LIBND4J_HHSEQUENCE_H
