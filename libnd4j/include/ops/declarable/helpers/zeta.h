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
// Created by Yurii Shyrma on 12.12.2017.
//

#ifndef LIBND4J_ZETA_H
#define LIBND4J_ZETA_H

#include <ops/declarable/helpers/helpers.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


	// calculate the Hurwitz zeta function for arrays
    void zeta(nd4j::LaunchContext * context, const NDArray& x, const NDArray& q, NDArray& output);

    	
	
	// calculate the Hurwitz zeta function for scalars
	// fast implementation, it is based on Euler-Maclaurin summation formula
    template <typename T>
    _CUDA_HD T zetaScalar(const T x, const T q) {

        const T machep =  1.11022302462515654042e-16;

        // FIXME: @raver119
        // expansion coeffZetaicients for Euler-Maclaurin summation formula (2k)! / B2k, where B2k are Bernoulli numbers
        const T coeffZeta[] = { 12.0,-720.0,30240.0,-1209600.0,47900160.0,-1.8924375803183791606e9,7.47242496e10,-2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15, 1.8152105401943546773e17, -7.1661652561756670113e18};

        // if (x <= (T)1.)
        // 	throw("zeta function: x must be > 1 !");

        // if (q <= (T)0.)
        // 	throw("zeta function: q must be > 0 !");

        T a, b(0.), k, s, t, w;

        s = math::nd4j_pow<T, T, T>(q, -x);
        a = q;
        int i = 0;

        while(i < 9 || a <= (T)9.) {
            i += 1;
            a += (T)1.0;
            b = math::nd4j_pow<T, T, T>(a, -x);
            s += b;
            if(math::nd4j_abs(b / s) < (T)machep)
                return s;
        }

        w = a;
        s += b * (w / (x - (T)1.) - (T)0.5);
        a = (T)1.;
        k = (T)0.;

        for(i = 0; i < 12; ++i) {
            a *= x + k;
            b /= w;
            t = a * b / coeffZeta[i];
            s += t;
            t = math::nd4j_abs(t / s);

            if(t < (T)machep)
                return s;

            k += (T)1.f;
            a *= x + k;
            b /= w;
            k += (T)1.f;
        }

        return s;
    }



}
}
}


#endif //LIBND4J_ZETA_H
