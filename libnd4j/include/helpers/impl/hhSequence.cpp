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

#include <helpers/hhSequence.h>
#include <helpers/householder.h>

namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
HHsequence::HHsequence(const NDArray& vectors, const NDArray& coeffs, const char type): _vectors(vectors), _coeffs(coeffs) {

	_diagSize = sd::math::nd4j_min(_vectors.sizeAt(0), _vectors.sizeAt(1));
	_shift = 0;
	_type  = type;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence::mulLeft_(NDArray& matrix) {

	const int rows   = _vectors.sizeAt(0);
	const int cols   = _vectors.sizeAt(1);
	const int inRows = matrix.sizeAt(0);

	for(int i = _diagSize - 1; i >= 0; --i) {

    	if(_type == 'u') {

    		NDArray block = matrix({inRows-rows+_shift+ i,inRows,  0,0}, true);
    		Householder<T>::mulLeft(block, _vectors({i + 1 + _shift, rows, i, i+1}, true), _coeffs.t<T>(i));
    	}
    	else {

    		NDArray block = matrix({inRows-cols+_shift+i,inRows,  0,0}, true);
    		Householder<T>::mulLeft(block, _vectors({i, i+1, i + 1 + _shift, cols}, true), _coeffs.t<T>(i));
    	}
    }
}


//////////////////////////////////////////////////////////////////////////
NDArray HHsequence::getTail(const int idx) const {


    int first = idx + 1 + _shift;

    if(_type == 'u')
        return _vectors({first, -1, idx, idx+1}, true);
    else
        return _vectors({idx, idx+1, first, -1}, true);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence::applyTo_(NDArray& dest) {

    int size = _type == 'u' ? _vectors.sizeAt(0) : _vectors.sizeAt(1);

    if(dest.rankOf() != 2 || (dest.sizeAt(0) != size && dest.sizeAt(1) != size))
        dest = NDArray(dest.ordering(), {size, size}, dest.dataType(), dest.getContext());
    dest.setIdentity();

    for(int k = _diagSize - 1; k >= 0; --k) {

        int curNum = size - k - _shift;
        if(curNum < 1 || (k + 1 + _shift) >= size )
            continue;
        auto block =  dest({dest.sizeAt(0)-curNum,dest.sizeAt(0),  dest.sizeAt(1)-curNum,dest.sizeAt(1)}, true);

        Householder<T>::mulLeft(block, getTail(k), _coeffs.t<T>(k));
    }
}

//////////////////////////////////////////////////////////////////////////
void HHsequence::applyTo(NDArray& dest) {
    auto xType = _coeffs.dataType();
    BUILD_SINGLE_SELECTOR(xType, applyTo_, (dest), FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
void HHsequence::mulLeft(NDArray& matrix) {
    auto xType = _coeffs.dataType();
    BUILD_SINGLE_SELECTOR(xType, mulLeft_, (matrix), FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void HHsequence::applyTo_, (sd::NDArray &dest), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void HHsequence::mulLeft_, (NDArray& matrix), FLOAT_TYPES);

}
}
}
