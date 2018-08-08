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

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> Householder<T>::evalHHmatrix(const NDArray<T>& x) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrix method: input array must be vector or scalar!");

	NDArray<T> w(x.ordering(),  {(int)x.lengthOf(), 1}, x.getWorkspace());							// column-vector
	NDArray<T> wT(x.ordering(), {1, (int)x.lengthOf()}, x.getWorkspace());							// row-vector (transposed w)	

	T coeff;
	T normX = x.template reduceNumber<simdOps::Norm2<T>>();		
	
	if(normX*normX - x(0.)*x(0.) <= DataTypeUtils::min<T>() || x.lengthOf() == 1) {

		normX = x(0.); 
		coeff = (T)0.;		
		w = (T)0.;
		
	} 	
	else {
		
		if(x(0.) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0.) - normX;
		coeff = -u0 / normX;				
		w.assign(x / u0);		
	}
	
	w(0.) = (T)1.;
	wT.assign(&w);
	
	NDArray<T> identity(x.ordering(), {(int)x.lengthOf(), (int)x.lengthOf()}, x.getWorkspace());					 
	identity.setIdentity();																			// identity matrix	

	return identity - mmul(w, wT) * coeff;	

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(const NDArray<T>& x, NDArray<T>& tail, T& coeff, T& normX) {

	// input validation
	if(!x.isVector() && !x.isScalar())
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input array must be vector or scalar!");

	if(!x.isScalar() && x.lengthOf() != tail.lengthOf() + 1)
		throw std::runtime_error("ops::helpers::Householder::evalHHmatrixData method: input tail vector must have length less than unity compared to input x vector!");

	normX = x.template reduceNumber<simdOps::Norm2<T>>();		
		
	if(normX*normX - x(0.)*x(0.) <= DataTypeUtils::min<T>() || x.lengthOf() == 1) {

		normX = x(0.);
		coeff = (T)0.;		
		tail = (T)0.;		
	}
	else {
		
		if(x(0.) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0.) - normX;
		coeff = -u0 / normX;				

		if(x.isRowVector())
			tail.assign(x({{}, {1, -1}}) / u0);		
		else
			tail.assign(x({{1, -1}, {}}) / u0);		
	}		
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(const NDArray<T>& x, T& coeff, T& normX) {

	int rows = (int)x.lengthOf()-1;
	int num = 1;
	
	if(rows == 0) {
		rows = 1;
		num = 0;
	}	
	
	NDArray<T> tail(x.ordering(), {rows, 1}, x.getWorkspace());
	evalHHmatrixData(x, tail, coeff, normX);

	if(x.isRowVector()) {
		NDArray<T>* temp = x.subarray({{}, {num, x.sizeAt(1)}});
		temp->assign(tail);
		delete temp;
	}
	else {		
		NDArray<T>* temp = x.subarray({{num, x.sizeAt(0)}, {}});
		temp->assign(tail);
		delete temp;
	}
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {
	
	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulLeft method: input array must be 2D matrix !";	

	if(matrix.sizeAt(0) == 1)   
    	matrix *= (T)1. - coeff;
  	
  	else if(coeff != (T)0.) {

  		NDArray<T>* bottomPart =  matrix.subarray({{1, matrix.sizeAt(0)}, {}});
		NDArray<T> bottomPartCopy = *bottomPart; 

		if(tail.isColumnVector()) {

			NDArray<T> column = tail;
			NDArray<T>* row = tail.transpose();						
    		NDArray<T> resultingRow = mmul(*row, bottomPartCopy);
    		NDArray<T>* fistRow = matrix.subarray({{0,1}, {}});
    		resultingRow += *fistRow;        	
    		*fistRow -= resultingRow * coeff;	
    		*bottomPart -= mmul(column, resultingRow) * coeff;    		

			delete row;
			delete fistRow;
		}
		else {
			
			NDArray<T> row = tail;
			NDArray<T>* column = tail.transpose();
    		NDArray<T> resultingRow = mmul(row, bottomPartCopy);
    		NDArray<T>* fistRow = matrix.subarray({{0,1}, {}});
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
void Householder<T>::mulRight(NDArray<T>& matrix, const NDArray<T>& tail, const T coeff) {

	// if(matrix.rankOf() != 2)
	// 	throw "ops::helpers::Householder::mulRight method: input array must be 2D matrix !";
	
	if(matrix.sizeAt(1) == 1)   
    	matrix *= (T)1. - coeff;
  	
  	else if(coeff != (T)0.) {

  		NDArray<T>* rightPart =  matrix.subarray({{}, {1, matrix.sizeAt(1)}});
		NDArray<T> rightPartCopy = *rightPart; 
		NDArray<T>* fistCol = matrix.subarray({{},{0,1}});

  		if(tail.isColumnVector()) {

			NDArray<T> column = tail;
			NDArray<T>* row = tail.transpose();						
    		NDArray<T> resultingCol = mmul(rightPartCopy, column);    		
    		resultingCol += *fistCol;        	
    		*fistCol -= resultingCol * coeff;	
    		*rightPart -= mmul(resultingCol, *row) * coeff;    		

			delete row;			
		}
		else {
			
			NDArray<T> row = tail;
			NDArray<T>* column = tail.transpose();
    		NDArray<T> resultingCol = mmul(rightPartCopy, *column);    		
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
template class ND4J_EXPORT Householder<int>;
template class ND4J_EXPORT Householder<Nd4jLong>;







}
}
}
