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

#include<ops/declarable/helpers/zeta.h>

namespace nd4j {
namespace ops {
namespace helpers {

const int maxIter = 1000000;							// max number of loop iterations

//////////////////////////////////////////////////////////////////////////
// slow implementation
template <typename T>
static FORCEINLINE T zetaScalarSlow(const T x, const T q) {
	
	const T precision = (T)1e-7; 									// function stops the calculation of series when next item is <= precision
		
	// if (x <= (T)1.) 
	// 	throw("zeta function: x must be > 1 !");

	// if (q <= (T)0.) 
	// 	throw("zeta function: q must be > 0 !");

	T item;
	T result = (T)0.;
	for(int i = 0; i < maxIter; ++i) {		
		
		item = math::nd4j_pow((q + i),-x);
		result += item;
		
		if(item <= precision)
			break;
	}

	return result;
}


//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function for arrays
template <typename T>
static void zeta_(graph::LaunchContext* context, const NDArray& x, const NDArray& q, NDArray* output) {

	//auto result = NDArray(&x, false, context);
	int xLen = x.lengthOf();

	PRAGMA_OMP_PARALLEL_FOR_IF(xLen > Environment::getInstance()->elementwiseThreshold())
	for(int i = 0; i < xLen; ++i)
		  z.p(i, zetaScalar<T>(x.e<T>(i), q.e<T>(i)));
}

void zeta(graph::LaunchContext* context, const NDArray& x, const NDArray& q, NDArray& z) {
    BUILD_SINGLE_SELECTOR(x.dataType(), zeta_, (context, x, q, z), FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void zeta_, (graph::LaunchContext* context, const NDArray& x, const NDArray& q, NDArray& z), FLOAT_TYPES);

}
}
}

