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
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    _CUDA_HD const int maxIter = 1000000;							// max number of loop iterations
    _CUDA_HD const double machep =  1.11022302462515654042e-16;

    // expansion coefficients for Euler-Maclaurin summation formula (2k)! / B2k, where B2k are Bernoulli numbers
    _CUDA_HD const double coeff[] = { 12.0,-720.0,30240.0,-1209600.0,47900160.0,-1.8924375803183791606e9,7.47242496e10,-2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15, 1.8152105401943546773e17, -7.1661652561756670113e18};


    //////////////////////////////////////////////////////////////////////////
    // fast implementation, it is based on Euler-Maclaurin summation formula
    template <typename T>
    T zeta(const T x, const T q) {
        return (T) 0;
    }

    //////////////////////////////////////////////////////////////////////////
    // calculate the Hurwitz zeta function for arrays
    template <typename T>
    static NDArray zeta_(const NDArray& x, const NDArray& q) {
	    auto result = NDArray(&x, false, x.getContext());

	    return result;
    }

	NDArray zeta(const NDArray& x, const NDArray& q) {
		BUILD_SINGLE_SELECTOR(x.dataType(), return zeta_, (x, q), FLOAT_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template NDArray zeta_, (const NDArray& x, const NDArray& q), FLOAT_TYPES);


    template bfloat16 zeta(bfloat16, bfloat16);
    template float16 zeta(float16, float16);
    template float zeta(float, float);
    template double zeta(double, double);
}
}
}

