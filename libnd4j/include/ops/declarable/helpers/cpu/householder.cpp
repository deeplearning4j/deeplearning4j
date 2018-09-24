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
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/helpers/householder.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray Householder<T>::evalHHmatrix(const NDArray& x) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrix method: input array must be vector or scalar!");

	auto w = NDArrayFactory::create(x.ordering(),  {(int)x.lengthOf(), 1}, x.dataType(), x.getWorkspace());							// column-vector
	auto wT = NDArrayFactory::create(x.ordering(), {1, (int)x.lengthOf()}, x.dataType(), x.getWorkspace());							// row-vector (transposed w)

	T coeff;
	T normX = x.reduceNumber(reduce::Norm2).e<T>(0);
	
	if(normX*normX - x.e<T>(0) * x.e<T>(0) <= DataTypeUtils::min<T>() || x.lengthOf() == 1) {

		normX = x.e<T>(0);
		coeff = 0.f;
		w = 0.f;
		
	} 	
	else {
		
		if(x.e<T>(0) >= (T)0.f)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x.e<T>(0) - normX;
		coeff = -u0 / normX;				
		w.assign(x / u0);		
	}
	
	w.p(Nd4jLong(0), 1.f);
	wT.assign(&w);
	
	auto identity = NDArrayFactory::create(x.ordering(), {(int)x.lengthOf(), (int)x.lengthOf()}, x.dataType(), x.getWorkspace());
	identity.setIdentity();																			// identity matrix	

	return identity - mmul(w, wT) * coeff;	

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(const NDArray& x, NDArray& tail, T& coeff, T& normX) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input array must be vector or scalar!");

	if(!x.isScalar() && x.lengthOf() != tail.lengthOf() + 1)
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input tail vector must have length less than unity compared to input x vector!");

	normX = x.reduceNumber(reduce::Norm2, nullptr).e<T>(0);
		
	if(normX*normX - x.e<T>(0) * x.e<T>(0) <= DataTypeUtils::min<T>() || x.lengthOf() == 1) {

		normX = x.e<T>(0);
		coeff = (T)0.f;
		tail = (T)0.f;
	}
	else {
		
		if(x.e<T>(0) >= (T)0.f)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x.e<T>(0) - normX;
		coeff = -u0 / normX;				

		if(x.isRowVector())
			tail.assign(x({0,0, 1,-1}) / u0);		
		else
			tail.assign(x({1,-1, 0,0,}) / u0);		
	}		
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(const NDArray& x, T& coeff, T& normX) {

	int rows = (int)x.lengthOf()-1;
	int num = 1;
	
	if(rows == 0) {
		rows = 1;
		num = 0;
	}	
	
	auto tail = NDArrayFactory::create(x.ordering(), {rows, 1}, x.dataType(), x.getWorkspace());
	evalHHmatrixData(x, tail, coeff, normX);

	if(x.isRowVector()) {
		auto temp = x.subarray({{}, {num, x.sizeAt(1)}});
		temp->assign(tail);
		delete temp;
	}
	else {		
		auto temp = x.subarray({{num, x.sizeAt(0)}, {}});
		temp->assign(tail);
		delete temp;
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray& matrix, const NDArray& tail, const T coeff) {
	
	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulLeft method: input array must be 2D matrix !";	

	if(matrix.sizeAt(0) == 1)   
    	matrix *= (T)1.f - coeff;
  	
  	else if(coeff != (T)0.f) {

  		auto bottomPart =  matrix.subarray({{1, matrix.sizeAt(0)}, {}});
		auto bottomPartCopy = *bottomPart;

		if(tail.isColumnVector()) {

			auto column = tail;
			auto row = tail.transpose();
    		auto resultingRow = mmul(*row, bottomPartCopy);
    		auto fistRow = matrix.subarray({{0,1}, {}});
    		resultingRow += *fistRow;        	
    		*fistRow -= resultingRow * coeff;	
    		*bottomPart -= mmul(column, resultingRow) * coeff;    		

			delete row;
			delete fistRow;
		}
		else {
			
			auto row = tail;
			auto column = tail.transpose();
    		auto resultingRow = mmul(row, bottomPartCopy);
    		auto fistRow = matrix.subarray({{0,1}, {}});
    		resultingRow += *fistRow;        	
    		*fistRow -= resultingRow * coeff;
    		*bottomPart -= mmul(*column, resultingRow) * coeff;    	

			delete column;
			delete fistRow;
		}	    	    	
		delete bottomPart;
	}
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulRight(NDArray& matrix, const NDArray& tail, const T coeff) {

	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulRight method: input array must be 2D matrix !";
	
	if(matrix.sizeAt(1) == 1)   
    	matrix *= (T)1.f - coeff;
  	
  	else if(coeff != (T)0.f) {

  		auto rightPart =  matrix.subarray({{}, {1, matrix.sizeAt(1)}});
		auto rightPartCopy = *rightPart;
		auto fistCol = matrix.subarray({{},{0,1}});

  		if(tail.isColumnVector()) {

			auto column = tail;
			auto row = tail.transpose();
    		auto resultingCol = mmul(rightPartCopy, column);
    		resultingCol += *fistCol;        	
    		*fistCol -= resultingCol * coeff;	
    		*rightPart -= mmul(resultingCol, *row) * coeff;    		

			delete row;			
		}
		else {
			
			auto row = tail;
			auto column = tail.transpose();
    		auto resultingCol = mmul(rightPartCopy, *column);
    		resultingCol += *fistCol;        	
    		*fistCol -= resultingCol * coeff;
    		*rightPart -= mmul(resultingCol, row) * coeff;

			delete column;
			
		}	    	    	
  		delete rightPart;
  		delete fistCol;
	}
}

      
template class ND4J_EXPORT Householder<float>;
template class ND4J_EXPORT Householder<float16>;
template class ND4J_EXPORT Householder<double>;







}
}
}
