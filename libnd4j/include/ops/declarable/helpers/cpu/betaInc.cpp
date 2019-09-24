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

///////////////////////////////////////////////////////////////////
// modified Lentz’s algorithm for continued fractions,
// reference: Lentz, W.J. 1976, “Generating Bessel Functions in Mie Scattering Calculations Using Continued Fractions”
template <typename T>
static T continuedFraction(const T a, const T b, const T x) {

	const T min = DataTypeUtils::min<T>() / DataTypeUtils::eps<T>();
    const T aPlusb = a + b;
    T val, delta, aPlus2i;

    // first iteration
    T c = 1;
    T d = static_cast<T>(1) - aPlusb * x / (a + static_cast<T>(1));
    if(math::nd4j_abs<T>(d) < min)
		d = min;
	d = static_cast<T>(1) / d;
    T f = d;

    for(uint i = 1; i <= maxIter; i += 2) {

    	aPlus2i = a + static_cast<T>(2*i);

		/***** even part *****/
		val = i * (b - i) * x / ((aPlus2i - static_cast<T>(1)) * aPlus2i);
		// d
		d = static_cast<T>(1) + val * d;
		if(math::nd4j_abs<T>(d) < min)
			d = min;
		d = static_cast<T>(1) / d;
		// c
		c = static_cast<T>(1) + val / c;
		if(math::nd4j_abs<T>(c) < min)
			c = min;
		// f
		f *= c * d;


		/***** odd part *****/
		val = -(a + i) * (aPlusb + i) * x / ((aPlus2i + static_cast<T>(1)) * aPlus2i);
		// d
		d = static_cast<T>(1) + val * d;
		if(math::nd4j_abs<T>(d) < min)
			d = min;
		d = static_cast<T>(1) / d;
		// c
		c = static_cast<T>(1) + val / c;
		if(math::nd4j_abs<T>(c) < min)
			c = min;
		// f
		delta = c * d;
		f *= delta;

		// condition to stop loop
		if(math::nd4j_abs<T>(delta - static_cast<T>(1)) <= DataTypeUtils::eps<T>())
			return f;
    }

    return 1.f / 0.f;	// no convergence, more iterations is required
}

///////////////////////////////////////////////////////////////////
// evaluates incomplete beta function for positive a and b, and x between 0 and 1.
template <typename T>
static T betaIncCore(T a, T b, T x) {
	// if (a <= (T)0. || b <= (T)0.)
	// 	throw("betaInc function: a and b must be > 0 !");

	// if (x < (T)0. || x > (T)1.)
	// 	throw("betaInc function: x must be within (0, 1) interval !");


	// t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
	if(a == b && x == static_cast<T>(0.5))
		return static_cast<T>(0.5);

	if (x == static_cast<T>(0) || x == static_cast<T>(1))
		return x;

	const T gammaPart = lgamma(a) + lgamma(b) - lgamma(a + b);
    const T front = math::nd4j_exp<T,T>(math::nd4j_log<T, T>(x) * a + math::nd4j_log<T, T>(1 - x) * b - gammaPart) / a;

	if (x <= (a + static_cast<T>(1)) / (a + b + static_cast<T>(2)))
		return front * continuedFraction(a, b, x);
	else // symmetry relation
		return static_cast<T>(1) - front * continuedFraction(b, a, static_cast<T>(1) - x);

}

///////////////////////////////////////////////////////////////////
template<typename T>
static void betaIncForArray(nd4j::LaunchContext * context, const NDArray& a, const NDArray& b, const NDArray& x, NDArray& output) {

	int xLen = x.lengthOf();

    PRAGMA_OMP_PARALLEL_FOR_IF(xLen > Environment::getInstance()->elementwiseThreshold())
	for(int i = 0; i < xLen; ++i)
		output.t<T>(i) = betaIncCore<T>(a.t<T>(i), b.t<T>(i), x.t<T>(i));
}

///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
void betaInc(nd4j::LaunchContext * context, const NDArray& a, const NDArray& b, const NDArray& x, NDArray& output) {
	auto xType = a.dataType();
	BUILD_SINGLE_SELECTOR(xType, betaIncForArray, (context, a, b, x, output), FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void betaIncForArray, (nd4j::LaunchContext * context, const NDArray& a, const NDArray& b, const NDArray& x, NDArray& output), FLOAT_TYPES);


}
}
}

