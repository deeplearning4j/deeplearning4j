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
// Created by Yurii Shyrma on 11.12.2017
//

#include<cmath> 
#include <DataTypeUtils.h>
#include<ops/declarable/helpers/betaInc.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

const int maxIter = 10000;				// max number of loop iterations in function for continued fractions 
const int maxValue = 3000;				// if a and b are both > maxValue, then apply Gauss-Legendre quadrature.


// 18 values of abscissas and weights for 36-point Gauss-Legendre integration,
// take a note - weights and abscissas are symmetric around the midpoint of the range of integration: 36/2 = 18
const double abscissas[18] = {0.0021695375159141994,
0.011413521097787704,0.027972308950302116,0.051727015600492421,
0.082502225484340941, 0.12007019910960293,0.16415283300752470,
0.21442376986779355, 0.27051082840644336, 0.33199876341447887,
0.39843234186401943, 0.46931971407375483, 0.54413605556657973,
0.62232745288031077, 0.70331500465597174, 0.78649910768313447,
0.87126389619061517, 0.95698180152629142};
const double weights[18] = {0.0055657196642445571,
0.012915947284065419,0.020181515297735382,0.027298621498568734,
0.034213810770299537,0.040875750923643261,0.047235083490265582,
0.053244713977759692,0.058860144245324798,0.064039797355015485,
0.068745323835736408,0.072941885005653087,0.076598410645870640,
0.079687828912071670,0.082187266704339706,0.084078218979661945,
0.085346685739338721,0.085983275670394821};




///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// modified Lentz’s algorithm for continued fractions, 
// reference: Lentz, W.J. 1976, “Generating Bessel Functions in Mie Scattering Calculations Using Continued Fractions,” 
template <typename T> 
static T continFract(const T a, const T b, const T x) {	

	const T min = DataTypeUtils::min<T>() / DataTypeUtils::eps<T>();
    const T amu = a - (T)1.;
    const T apu = a + (T)1.;
    const T apb = a + b;

    // first iteration 
    T coeff1 = (T)1.;
    T coeff2 = (T)1. - apb * x / apu; 
    if(math::nd4j_abs<T>(coeff2) < min)
			coeff2 = min;
	coeff2 = (T)1./coeff2;
    T result = coeff2;
         
    T val, delta;
    int i2;
    // rest iterations
    for(int i=1; i <= maxIter; i+=2) {        
    	i2 = 2*i;
		
		// even step
		val = i * (b - (T)i) * x / ((amu + (T)i2) * (a + (T)i2));		

		coeff2 = (T)(1.) + val * coeff2;
		if(math::nd4j_abs<T>(coeff2) < min)
			coeff2 = min;
		coeff2 = (T)1. / coeff2;

		coeff1 = (T)(1.) + val / coeff1;
		if(math::nd4j_abs<T>(coeff1) < min)
			coeff1 = min;		
		
		result *= coeff1 * coeff2;

		//***********************************************//
		// odd step
		val = -(a + (T)i) * (apb + (T)i) * x / ((a + (T)i2) * (apu + (T)i2));
		
		coeff2 = (T)(1.) + val * coeff2;
		if(math::nd4j_abs<T>(coeff2) < min)
			coeff2 = min;
		coeff2 = (T)1. / coeff2;

		coeff1 = (T)(1.) + val / coeff1;
		if(math::nd4j_abs<T>(coeff1) < min)
			coeff1 = min;

		delta = coeff1 * coeff2;
		result *= delta;
		
		// condition to stop loop		
		if(math::nd4j_abs<T>(delta - (T)1.) <= DataTypeUtils::eps<T>()) 		
			break;
    }
    
    return result;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// evaluates incomplete beta integral using Gauss-Legendre quadrature method
template <typename T>
static T gausLegQuad(const T a, const T b, const T x) {

	T upLim, t, result;
	T sum = (T)0.;
	T amu = a - (T)1.; 
	T bmu = b - (T)1.;
	T rat = a / (a + b);
	T lnrat  = math::nd4j_log<T,T>(rat);
	T lnratm = math::nd4j_log<T,T>((T)1. - rat);

	t = math::nd4j_sqrt<T,T>(a * b /((a + b) * (a + b) * (a + b + (T)1.)));
	if (x > rat) {	
		if (x >= (T)1.) 
			return (T)1.0;
		upLim = math::nd4j_min<T>((T)1., math::nd4j_max<T>(rat + (T)1.*t, x + (T)5.*t));
	} 
	else {	
		if (x <= (T)0.) 
			return (T)0.;
		upLim = math::nd4j_max<T>(0., math::nd4j_min<T>(rat - (T)10.*t, x - (T)5.*t));
	}	

	// Gauss-Legendre
#pragma omp declare reduction (add : double,float,float16 : omp_out += omp_in) initializer(omp_priv = (T)0.)
#pragma omp simd private(t) reduction(add:sum)
	for (int i = 0; i < 18; ++i) {	
		t = x + (upLim - x) * (T)abscissas[i];
		sum += (T)weights[i] * math::nd4j_exp<T,T>(amu * (math::nd4j_log<T,T>(t) - lnrat) + bmu * (math::nd4j_log<T,T>((T)1. - t) - lnratm));
	}
	result = sum * (upLim - x) * math::nd4j_exp<T,T>(amu * lnrat - lgamma(a) + bmu * lnratm - lgamma(b) + lgamma(a + b));
	
	if(result > (T)0.)
		return (T)1. - result;

	return -result;
}


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// evaluates incomplete beta function for positive a and b, and x between 0 and 1.
template <typename T> 
static T betaIncTA(T a, T b, T x) {
	// if (a <= (T)0. || b <= (T)0.) 
	// 	throw("betaInc function: a and b must be > 0 !");

	// if (x < (T)0. || x > (T)1.) 
	// 	throw("betaInc function: x must be within (0, 1) interval !");
	

	// t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
	if(a == b && x == (T)0.5)
		return (T)0.5;

	if (x == (T)0. || x == (T)1.) 
		return x;
	
	if (a > (T)maxValue && b > (T)maxValue) 
		return gausLegQuad<T>(a, b, x);	

	T front = math::nd4j_exp<T,T>( lgamma(a + b) - lgamma(a) - lgamma(b) + a * math::nd4j_log<T, T>(x) + b * math::nd4j_log<T, T>((T)1. - x));
	
	// continued fractions
	if (x < (a + (T)1.) / (a + b + (T)2.)) 
		return front * continFract(a, b, x) / a;
	// symmetry relation
	else 
		return (T)1. - front * continFract(b, a, (T)1. - x) / b;

}

template<typename T>
NDArray betaIncT(const NDArray& a, const NDArray& b, const NDArray& x) {
	auto result = NDArrayFactory::_create(&x, false, x.getWorkspace());

#pragma omp parallel for if(x.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
	for(int i = 0; i < x.lengthOf(); ++i) {
		result.p(i, betaIncTA<T>(a.e<T>(i), b.e<T>(i), x.e<T>(i)));
	}

	return result;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
NDArray betaInc(const NDArray& a, const NDArray& b, const NDArray& x) {
	auto xType = a.dataType();
	BUILD_SINGLE_SELECTOR(xType, return betaIncT, (a, b, x), FLOAT_TYPES);
}


template float   continFract<float>  (const float   a, const float   b, const float   x);
template float16 continFract<float16>(const float16 a, const float16 b, const float16 x);
template double  continFract<double> (const double  a, const double  b, const double  x);

template float   gausLegQuad<float>  (const float   a, const float   b, const float   x);
template float16 gausLegQuad<float16>(const float16 a, const float16 b, const float16 x);
template double  gausLegQuad<double> (const double  a, const double  b, const double  x);

template float   betaIncTA<float>  (const float   a, const float   b, const float   x);
template float16 betaIncTA<float16>(const float16 a, const float16 b, const float16 x);
template double  betaIncTA<double> (const double  a, const double  b, const double  x);

template NDArray betaIncT<float>  (const NDArray&   a, const NDArray&   b, const NDArray&  x);
template NDArray betaIncT<float16>(const NDArray& a, const NDArray& b, const NDArray& x);
template NDArray betaIncT<double> (const NDArray&  a, const NDArray&  b, const NDArray& x);


}
}
}

