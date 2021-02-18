/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by Yurii Shyrma on 18.12.2017
//


#include <helpers/householder.h>
#include <helpers/biDiagonalUp.h>


namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
BiDiagonalUp::BiDiagonalUp(const NDArray& matrix): _HHmatrix(NDArray(matrix.ordering(), {matrix.sizeAt(0), matrix.sizeAt(1)}, matrix.dataType(), matrix.getContext())),
                                                   _HHbidiag(NDArray(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.dataType(), matrix.getContext())) {

	// input validation
	if(matrix.rankOf() != 2 || matrix.isScalar())
		throw std::runtime_error("ops::helpers::biDiagonalizeUp constructor: input array must be 2D matrix !");

	_HHmatrix.assign(&matrix);
	_HHbidiag.assign(0.);

	evalData();
}

template <typename T>
void BiDiagonalUp::_evalData() {

	const auto rows = _HHmatrix.sizeAt(0);
	const auto cols = _HHmatrix.sizeAt(1);

	if(rows < cols)
		throw std::runtime_error("ops::helpers::BiDiagonalizeUp::evalData method: this procedure is applicable only for input matrix with rows >= cols !");

	T coeff, normX;

	T x, y;

	for(Nd4jLong i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix nullifying columns
		NDArray column1 = _HHmatrix({i,rows,  i,i+1});

        x = _HHmatrix.t<T>(i,i);
        y = _HHbidiag.t<T>(i,i);

		Householder<T>::evalHHmatrixDataI(column1, x, y);

        _HHmatrix.r<T>(i, i) = x;
        _HHbidiag.r<T>(i, i) = y;

		// multiply corresponding matrix block on householder matrix from the left: P * bottomRightCorner
		NDArray bottomRightCorner1 = _HHmatrix({i,rows,  i+1,cols}, true);	// {i, cols}
		Householder<T>::mulLeft(bottomRightCorner1, _HHmatrix({i+1,rows, i,i+1}, true), _HHmatrix.t<T>(i,i));

		if(i == cols-2)
			continue; 										// do not apply right multiplying at last iteration

		// evaluate Householder matrix nullifying rows
		NDArray row1 = _HHmatrix({i,i+1,  i+1,cols});

        x = _HHmatrix.t<T>(i,i+1);
        y = _HHbidiag.t<T>(i,i+1);

		Householder<T>::evalHHmatrixDataI(row1, x, y);

        _HHmatrix.r<T>(i, i+1) = x;
        _HHbidiag.r<T>(i, i+1) = y;

		// multiply corresponding matrix block on householder matrix from the right: bottomRightCorner * P
		NDArray bottomRightCorner2 = _HHmatrix({i+1,rows,  i+1,cols}, true);  // {i, rows}

		Householder<T>::mulRight(bottomRightCorner2, _HHmatrix({i,i+1, i+2,cols}, true), _HHmatrix.t<T>(i,i+1));
	}

	NDArray row2 =_HHmatrix({cols-2,cols-1, cols-1,cols});

	x = _HHmatrix.t<T>(cols-2,cols-1);
	y = _HHbidiag.t<T>(cols-2,cols-1);

	Householder<T>::evalHHmatrixDataI(row2, x, y);

    _HHmatrix.r<T>(cols-2,cols-1) = x;
    _HHbidiag.r<T>(cols-2,cols-1) = y;

	NDArray column2 = _HHmatrix({cols-1,rows, cols-1,cols});

	x = _HHmatrix.t<T>(cols-1,cols-1);
	y = _HHbidiag.t<T>(cols-1,cols-1);

	Householder<T>::evalHHmatrixDataI(column2, x, y);

	_HHmatrix.r<T>(cols-1, cols-1) = x;
    _HHbidiag.r<T>(cols-1, cols-1) = y;
}

//////////////////////////////////////////////////////////////////////////
void BiDiagonalUp::evalData() {
	auto xType = _HHmatrix.dataType();
	BUILD_SINGLE_SELECTOR(xType, _evalData, ();, FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence BiDiagonalUp::makeHHsequence_(const char type) {

	const int diagSize = type == 'u' ? _HHbidiag.sizeAt(0) : _HHbidiag.sizeAt(0) - 1;

	_hhCoeffs = NDArray(_HHmatrix.ordering(),  {diagSize}, _HHmatrix.dataType(), _HHmatrix.getContext());

	if(type == 'u')
	    for(int i = 0; i < diagSize; ++i)
	        _hhCoeffs.r<T>(i) = _HHmatrix.t<T>(i,i);
    else
    	for(int i = 0; i < diagSize; ++i)
        	_hhCoeffs.r<T>(i) = _HHmatrix.t<T>(i,i+1);

    HHsequence result(_HHmatrix, _hhCoeffs, type);

    if(type != 'u') {
        result._diagSize = diagSize;
    	result._shift  = 1;
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////
HHsequence BiDiagonalUp::makeHHsequence(const char type) {
	auto xType = _HHmatrix.dataType();
	BUILD_SINGLE_SELECTOR(xType, return makeHHsequence_, (type);, FLOAT_TYPES);
}



BUILD_SINGLE_TEMPLATE(template void BiDiagonalUp::_evalData, (), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template HHsequence BiDiagonalUp::makeHHsequence_, (const char type), FLOAT_TYPES);

}
}
}