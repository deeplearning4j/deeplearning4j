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

#include <ops/declarable/helpers/hhSequence.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence<T>::HHsequence(const NDArray<T>& vectors, const NDArray<T>& coeffs, const char type): _vectors(vectors), _coeffs(coeffs) {
	
	_diagSize = nd4j::math::nd4j_min(_vectors.sizeAt(0), _vectors.sizeAt(1));
	_shift = 0;    
	_type  = type;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence<T>::mulLeft(NDArray<T>& matrix) const {    		

	const int rows   = _vectors.sizeAt(0);
	const int cols   = _vectors.sizeAt(1);
	const int inRows = matrix.sizeAt(0);	

	NDArray<T>* block(nullptr);

	for(int i = _diagSize - 1; i >= 0; --i) {		
    	
    	if(_type == 'u') {
    		
    		block = matrix.subarray({{inRows - rows + _shift + i, inRows}, {}});
    		Householder<T>::mulLeft(*block, _vectors({{i + 1 + _shift, rows}, {i, i+1}}, true), _coeffs(i));
    	}
    	else {

    		block = matrix.subarray({{inRows - cols + _shift + i, inRows}, {}});
    		Householder<T>::mulLeft(*block, _vectors({{i, i+1}, {i + 1 + _shift, cols}}, true), _coeffs(i));    	
    	}

    	delete block;
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> HHsequence<T>::getTail(const int idx) const {

    
    int first = idx + 1 + _shift;
    
    if(_type == 'u')
        return _vectors({{first, -1},{idx, idx+1}}, true);
    else
        return _vectors({{idx, idx+1},{first, -1}}, true);    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence<T>::applyTo(NDArray<T>& dest) const{
    
    int size = _type == 'u' ? _vectors.sizeAt(0) : _vectors.sizeAt(1);

    if(dest.rankOf() != 2 || (dest.sizeAt(0) != size && dest.sizeAt(1) != size))
        dest = NDArray<T>(dest.ordering(), {size, size}, dest.getWorkspace());
    dest.setIdentity();
    
    for(int k = _diagSize - 1; k >= 0; --k) {
        
        int curNum = size - k - _shift;
        if(curNum < 1 || (k + 1 + _shift) >= size )
            continue;
        NDArray<T>* block = dest.subarray({{dest.sizeAt(0)-curNum, dest.sizeAt(0)},{dest.sizeAt(1)-curNum, dest.sizeAt(1)}});
        Householder<T>::mulLeft(*block, getTail(k), _coeffs(k));      
        delete block;
    }  
}


template class ND4J_EXPORT HHsequence<float>;
template class ND4J_EXPORT HHsequence<float16>;
template class ND4J_EXPORT HHsequence<double>;







}
}
}
