//
// Created by Yurii Shyrma on 18.12.2017
//

#include <ops/declarable/CustomOperations.h>
#include <DataTypeUtils.h>
#include <ops.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
void evalHouseholderData(const NDArray<T>& x, NDArray<T>& tail, T& normX, T& coeff) {

	// input validation
	if(!x.isVector())
		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
	
	if(!tail.isVector())
		throw "ops::helpers::houseHolderForVector function: output array must be vector !";

	if(x.lengthOf() != tail.lengthOf() + 1)
		throw "ops::helpers::houseHolderForVector function: output vector must have length smaller by unity compared to input vector !";
		
	normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();

	if(normX*normX - x(0)*x(0) <= min) {

		coeff = (T)0.;
		normX = x(0); 
		tail = (T)0.;
	}
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		const T u0 = x(0) - normX;
		coeff = -u0 / normX;
		
		if(x.isRowVector())
			tail = x({{}, {1,-1}}) / u0;
		else
			tail = x({{1,-1}, {}}) / u0;
	} 
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T> evalHouseholderMatrix(const NDArray<T>& x) {

	// input validation
	if(!x.isVector())
		throw "ops::helpers::houseHolderForVector function: input array must be vector !";
			
	NDArray<T> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());							// column-vector
	NDArray<T> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());							// row-vector (transposed w)	

	T normX = x.template reduceNumber<simdOps::Norm2<T>>();	
	const T min = DataTypeUtils::min<T>();
	T coeff;

	if(normX*normX - x(0)*x(0) <= min) {

		coeff = (T)0.;
		normX = x(0); 
		w = (T)0.;
	}
	else {
		
		if(x(0) >= (T)0.)
			normX = -normX;									// choose opposite sign to lessen roundoff error
		
		T u0 = x(0) - normX;
		coeff = -u0 / normX;				
		w.assign(x / u0);
	}

	w(0) = (T)1.;
	wT.assign(w);

	NDArray<T> identity((int)x.lengthOf(), (int)x.lengthOf(), x.ordering(), x.getWorkspace());					 
	identity.setIdentity();																			// identity matrix	

	return (identity - mmul(w, wT) * coeff);
}


template void evalHouseholderData<float>  (const NDArray<float  >& x, NDArray<float  >& tail, float  & normX, float  & coeff);
template void evalHouseholderData<float16>(const NDArray<float16>& x, NDArray<float16>& tail, float16& normX, float16& coeff);
template void evalHouseholderData<double> (const NDArray<double >& x, NDArray<double >& tail, double & normX, double & coeff);

template NDArray<float>   evalHouseholderMatrix<float>  (const NDArray<float  >& x);
template NDArray<float16> evalHouseholderMatrix<float16>(const NDArray<float16>& x);
template NDArray<double>  evalHouseholderMatrix<double> (const NDArray<double >& x);

}
}
}