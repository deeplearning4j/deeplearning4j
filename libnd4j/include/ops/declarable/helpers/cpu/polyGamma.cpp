//
// Created by Yurii Shyrma on 12.12.2017
//

#include<ops/declarable/helpers/polyGamma.h>
#include<ops/declarable/helpers/zeta.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// calculate factorial
template <typename T>
static T getFactorial(const int n) {
	if (n < 0)
		throw std::runtime_error("factorial is not defined for negative number !");

	if(n==0 || n==1)
		return (T)1.;

	T result = (T)1.;

#pragma omp declare reduction (dot : double,float,float16 : omp_out *= omp_in) initializer(omp_priv = (T)1.)
#pragma omp parallel for reduction(dot : result) schedule(static)
	for(int i = 2; i <= n; ++i)
		result *= i;
	
	return result;
}

//////////////////////////////////////////////////////////////////////////
// implementation is based on serial representation written in terms of the Hurwitz zeta function as polygamma = (-1)^{n+1} * n! * zeta(n+1, x)
template <typename T>
static T polyGamma(const int n, const T x) {
	
	// if (n < 0) 
	// 	throw("polyGamma function: n must be >= 0 !");

	// if (x <= (T)0.) 
	// 	throw("polyGamma function: x must be > 0 !");
	
	// TODO case for n = 0 (digamma)

	int sign = (n + 1) % 2  ?  -1 : 1;
	// T factorial = (T)std::tgamma(n + 1);		

	return sign * getFactorial<T>(n) * zeta<T>((T)(n + 1), x);	
}


//////////////////////////////////////////////////////////////////////////
// calculate polygamma function for arrays
template <typename T>
NDArray<T> polyGamma(const NDArray<T>& n, const NDArray<T>& x) {

	NDArray<T> result(&x, false, x.getWorkspace());

#pragma omp parallel for if(x.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)	
	for(int i = 0; i < x.lengthOf(); ++i)
		result(i) = polyGamma<T>((int)n(i), x(i));

	return result;
}


template float   polyGamma<float>  (const int n, const float   x);
template float16 polyGamma<float16>(const int n, const float16 x);
template double  polyGamma<double> (const int n, const double  x);

template NDArray<float>   polyGamma<float>  (const NDArray<float>&   n, const NDArray<float>&   x);
template NDArray<float16> polyGamma<float16>(const NDArray<float16>& n, const NDArray<float16>& x);
template NDArray<double>  polyGamma<double> (const NDArray<double>&  n, const NDArray<double>&  x);


}
}
}

