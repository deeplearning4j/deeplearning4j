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

#include<ops/declarable/helpers/gammaMathFunc.h>
#include<ops/declarable/helpers/zeta.h>
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>

namespace sd {
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

	for(int i = 2; i <= n; ++i)
		result *= i;

	return result;
}

//////////////////////////////////////////////////////////////////////////
// implementation is based on serial representation written in terms of the Hurwitz zeta function as polygamma = (-1)^{n+1} * n! * zeta(n+1, x)
template <typename T>
static FORCEINLINE T polyGammaScalar(sd::LaunchContext * context, const int n, const T x) {

	// if (n < 0)
	// 	throw("polyGamma function: n must be >= 0 !");

	// if (x <= (T)0.)
	// 	throw("polyGamma function: x must be > 0 !");

	int sign = (n + 1) % 2  ?  -1 : 1;
	// T factorial = (T)std::tgamma(n + 1);

	return sign * getFactorial<T>(n) * zetaScalar<T>((T)(n + 1), x);
}


//////////////////////////////////////////////////////////////////////////
// calculate polygamma function for arrays
template <typename T>
static void polyGamma_(sd::LaunchContext * context, const NDArray& n, const NDArray& x, NDArray& output) {

	auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
        	const T order = n.e<T>(i);
        	if(order != static_cast<int>(order))						// if order has fractional part then do not perform calculations and return NAN
        		output.p(i, std::numeric_limits<T>::quiet_NaN());
        	else if (order == 0)										// polygamma function of zero order is digamma function
        		output.p(i, diGammaScalar<T>(x.e<T>(i)));
        	else
            	output.p(i, polyGammaScalar<T>(context, order, x.e<T>(i)));
        }
    };
	samediff::Threads::parallel_for(func, 0, x.lengthOf());
}

	void polyGamma(sd::LaunchContext * context, const NDArray& n, const NDArray& x, NDArray& output) {
		BUILD_SINGLE_SELECTOR(x.dataType(), polyGamma_, (context, n, x, output), FLOAT_TYPES);
	}

BUILD_SINGLE_TEMPLATE(template void polyGamma_, (sd::LaunchContext * context, const NDArray& n, const NDArray& x, NDArray& output), FLOAT_TYPES);



}
}
}

