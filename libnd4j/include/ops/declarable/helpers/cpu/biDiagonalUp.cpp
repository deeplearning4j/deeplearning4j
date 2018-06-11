//
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/helpers/householder.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
BiDiagonalUp<T>::BiDiagonalUp(const NDArray<T>& matrix): _HHmatrix(NDArray<T>(matrix.ordering(), {matrix.sizeAt(0), matrix.sizeAt(1)}, matrix.getWorkspace())),  
                                                         _HHbidiag(NDArray<T>(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.getWorkspace())) {

	// input validation
	if(matrix.rankOf() != 2 || matrix.isScalar())
		throw std::runtime_error("ops::helpers::biDiagonalizeUp constructor: input array must be 2D matrix !");

	_HHmatrix.assign(&matrix);
	_HHbidiag.assign(0.);
	
	evalData();

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void BiDiagonalUp<T>::evalData() {
	
	const auto rows = _HHmatrix.sizeAt(0);
	const auto cols = _HHmatrix.sizeAt(1);
	
	if(rows < cols)
		throw std::runtime_error("ops::helpers::BiDiagonalizeUp::evalData method: this procedure is applicable only for input matrix with rows >= cols !");
		
	NDArray<T>* bottomRightCorner(nullptr), *column(nullptr), *row(nullptr);	
	T coeff, normX;	
	
	for(int i = 0; i < cols-1; ++i ) {

		// evaluate Householder matrix nullifying columns 		
		column = _HHmatrix.subarray({{i,   rows}, {i, i+1}});						
		Householder<T>::evalHHmatrixDataI(*column, _HHmatrix(i,i), _HHbidiag(i,i)); 
		// multiply corresponding matrix block on householder matrix from the left: P * bottomRightCorner		
		bottomRightCorner =  _HHmatrix.subarray({{i, rows}, {i+1, cols}});	// {i, cols}				
		Householder<T>::mulLeft(*bottomRightCorner, _HHmatrix({{i+1, rows}, {i, i+1}}, true), _HHmatrix(i,i));		

		delete bottomRightCorner;
		delete column;
		
		if(i == cols-2)			
			continue; 										// do not apply right multiplying at last iteration		

		// evaluate Householder matrix nullifying rows 
		row  = _HHmatrix.subarray({{i, i+1}, {i+1, cols}});
		Householder<T>::evalHHmatrixDataI(*row, _HHmatrix(i,i+1), _HHbidiag(i,i+1));				
		// multiply corresponding matrix block on householder matrix from the right: bottomRightCorner * P
		bottomRightCorner = _HHmatrix.subarray({{i+1, rows}, {i+1, cols}});  // {i, rows}		
		Householder<T>::mulRight(*bottomRightCorner, _HHmatrix({{i, i+1}, {i+2, cols}}, true), _HHmatrix(i,i+1));
	
		delete bottomRightCorner;
		delete row;
	}	

	row  = _HHmatrix.subarray({{cols-2, cols-1}, {cols-1, cols}});	
	Householder<T>::evalHHmatrixDataI(*row, _HHmatrix(cols-2,cols-1), _HHbidiag(cols-2,cols-1)); 
	delete row;	

	column = _HHmatrix.subarray({{cols-1, rows}, {cols-1, cols}});
	Householder<T>::evalHHmatrixDataI(*column, _HHmatrix(cols-1,cols-1), _HHbidiag(cols-1,cols-1)); 
	delete column;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHsequence<T> BiDiagonalUp<T>::makeHHsequence(const char type) const {

	if(type == 'u') {

    	const int diagSize = _HHbidiag.sizeAt(0);
    	NDArray<T> colOfCoeffs(_HHmatrix.ordering(),  {diagSize, 1}, _HHmatrix.getWorkspace());

	    for(int i = 0; i < diagSize; ++i)
	        colOfCoeffs(i) = _HHmatrix(i,i);

    	return HHsequence<T>(_HHmatrix, colOfCoeffs, type);
    }
    else {

    	const int diagUpSize = _HHbidiag.sizeAt(0) - 1;
		NDArray<T> colOfCoeffs(_HHmatrix.ordering(), {diagUpSize, 1}, _HHmatrix.getWorkspace());

    	for(int i = 0; i < diagUpSize; ++i)
        	colOfCoeffs(i) = _HHmatrix(i,i+1);

    	HHsequence<T> result(_HHmatrix, colOfCoeffs, type);
    	result._diagSize = diagUpSize;
    	result._shift  = 1;

    	return result;	
    }
}





template class ND4J_EXPORT BiDiagonalUp<float>;
template class ND4J_EXPORT BiDiagonalUp<float16>;
template class ND4J_EXPORT BiDiagonalUp<double>;



}
}
}