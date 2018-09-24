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
// Created by Yurii Shyrma on 12.12.2017
//

#include<ops/declarable/helpers/polyGamma.h>
#include<ops/declarable/helpers/zeta.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// calculate factorial
template <typename T>
static FORCEINLINE T getFactorial(const int n) {
	if (n < 0)
		throw std::runtime_error("factorial is not defined for negative number !");

	if(n==0 || n==1)
		return (T)1.f;

	T result = (T)1.f;

#pragma omp declare reduction (dot : double,float,float16 : omp_out *= omp_in) initializer(omp_priv = (T)1.)
#pragma omp parallel for reduction(dot : result) schedule(static)
	for(int i = 2; i <= n; ++i)
		result *= i;
	
	return result;
}

//////////////////////////////////////////////////////////////////////////
// implementation is based on serial representation written in terms of the Hurwitz zeta function as polygamma = (-1)^{n+1} * n! * zeta(n+1, x)
template <typename T>
static FORCEINLINE T polyGamma(const int n, const T x) {
	
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
static NDArray _polyGamma(const NDArray& n, const NDArray& x) {

	NDArray result = NDArrayFactory::_create(&x, false, x.getWorkspace());

#pragma omp parallel for if(x.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)	
	for(int i = 0; i < x.lengthOf(); ++i)
		result.putScalar(i, polyGamma<T>(n.e<int>(i), x.e<T>(i)));

	return result;
}

	NDArray polyGamma(const NDArray& n, const NDArray& x) {
		BUILD_SINGLE_SELECTOR(x.dataType(), _polyGamma, (n, x), FLOAT_TYPES);
	}

BUILD_SINGLE_TEMPLATE(template NDArray _polyGamma, (const NDArray&   n, const NDArray& x), FLOAT_TYPES);



}
}
}

