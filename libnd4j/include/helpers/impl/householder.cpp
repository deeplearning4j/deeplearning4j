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

namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// template <typename T>
// NDArray Householder<T>::evalHHmatrix(const NDArray& x) {

// 	// input validation
// 	if(x.rankOf() != 1 && !x.isScalar())
// 		throw std::runtime_error("ops::helpers::Householder::evalHHmatrix method: iinput array must have rank = 1 or to be scalar!");

// 	const auto xLen = x.lengthOf();

// 	NDArray w(x.ordering(), {xLen, 1}, x.dataType(), x.getContext());							// column-vector

// 	NDArray xTail = xLen > 1 ? x({1,-1}) : NDArray();
// 	T tailXnorm   = xLen > 1 ? xTail.reduceNumber(reduce::SquaredNorm).t<T>(0) : (T)0;

// 	const auto xFirstElem = x.t<T>(0);

// 	T coeff, normX;

// 	if(tailXnorm <= DataTypeUtils::min<T>()) {

// 		normX = xFirstElem;
// 		coeff = 0.f;
// 		if(xLen > 1)
// 			w({1,-1, 0,0}) = 0.f;
// 	}
// 	else {

// 		normX = math::nd4j_sqrt<T,T>(xFirstElem*xFirstElem + tailXnorm);

// 		if(xFirstElem >= (T)0.f)
// 			normX = -normX;									// choose opposite sign to lessen roundoff error

// 		coeff = (normX - xFirstElem) / normX;

// 		if(xLen > 1)
// 			w({1,-1, 0,0}).assign(xTail / (xFirstElem - normX));
// 	}

// 	w.t<T>(0) = (T)1;

// 	NDArray identity(x.ordering(), {xLen, xLen}, x.dataType(), x.getContext());
// 	identity.setIdentity();																			// identity matrix

// 	return identity - mmul(w, w.transpose()) * coeff;
// }

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(const NDArray& x, NDArray& tail, T& coeff, T& normX) {

	// input validation
	if(x.rankOf() != 1 && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input array must have rank = 1 or to be scalar!");

	if(!x.isScalar() && x.lengthOf() != tail.lengthOf() + 1)
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input tail vector must have length less than unity compared to input x vector!");

	const auto xLen = x.lengthOf();

	const NDArray xTail = xLen > 1 ? x({1,-1}) : NDArray();

	T tailXnorm   = xLen > 1 ? xTail.reduceNumber(reduce::SquaredNorm).t<T>(0) : (T)0;

	const auto xFirstElem = x.t<T>(0);

	if(tailXnorm <= DataTypeUtils::min<T>()) {

		normX = xFirstElem;
		coeff = (T)0.f;
		tail = (T)0.f;
	}
	else {

		normX = math::nd4j_sqrt<T,T>(xFirstElem*xFirstElem + tailXnorm);

		if(xFirstElem >= (T)0.f)
			normX = -normX;									// choose opposite sign to lessen roundoff error

		coeff = (normX - xFirstElem) / normX;

		tail.assign(xTail / (xFirstElem - normX));
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(NDArray& x, T& coeff, T& normX) {

	// input validation
	if(x.rankOf() != 1 && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixDataI method: input array must have rank = 1 or to be scalar!");

	int rows = (int)x.lengthOf()-1;
	int num = 1;

	if(rows == 0) {
		rows = 1;
		num = 0;
	}

	NDArray tail = x({num, -1});

	evalHHmatrixData(x, tail, coeff, normX);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray& matrix, const NDArray& tail, const T coeff) {

	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulLeft method: input array must be 2D matrix !";

	if(matrix.sizeAt(0) == 1 && coeff != (T)0) {

		matrix *= (T) 1.f - coeff;
    }
    else if(coeff != (T)0.f) {

  		NDArray bottomPart = matrix({1,matrix.sizeAt(0), 0,0}, true);
  		NDArray fistRow = matrix({0,1, 0,0}, true);

		if(tail.isColumnVector()) {

    		auto resultingRow = mmul(tail.transpose(), bottomPart);
    		resultingRow += fistRow;
    		resultingRow *= coeff;
    		fistRow -= resultingRow;
    		bottomPart -= mmul(tail, resultingRow);
		}
		else {

    		auto resultingRow = mmul(tail, bottomPart);
    		resultingRow += fistRow;
    		resultingRow *= coeff;
    		fistRow -= resultingRow;
    		bottomPart -= mmul(tail.transpose(), resultingRow);
		}
	}
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulRight(NDArray& matrix, const NDArray& tail, const T coeff) {

	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulRight method: input array must be 2D matrix !";

	if(matrix.sizeAt(1) == 1 && coeff != (T)0) {
    	matrix *= (T)1.f - coeff;
	}
  	else if(coeff != (T)0.f) {

  		NDArray rightPart = matrix({0,0, 1,matrix.sizeAt(1)}, true);
		NDArray fistCol   = matrix({0,0, 0,1}, true);

  		if(tail.isColumnVector()) {

    		auto resultingCol = mmul(rightPart, tail);
    		resultingCol += fistCol;
    		resultingCol *= coeff;
    		fistCol -= resultingCol;
    		rightPart -= mmul(resultingCol, tail.transpose());
		}
		else {

    		auto resultingCol = mmul(rightPart, tail.transpose());
    		resultingCol += fistCol;
    		resultingCol *= coeff;
    		fistCol -= resultingCol;
    		rightPart -= mmul(resultingCol, tail);
		}
	}
}


template class ND4J_EXPORT Householder<float>;
template class ND4J_EXPORT Householder<float16>;
template class ND4J_EXPORT Householder<bfloat16>;
template class ND4J_EXPORT Householder<double>;







}
}
}
