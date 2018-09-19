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

const int maxIter = 1000000;							// max number of loop iterations 
const double machep =  1.11022302462515654042e-16;		

// expansion coefficients for Euler-Maclaurin summation formula (2k)! / B2k, where B2k are Bernoulli numbers
const double coeff[] = { 12.0,-720.0,30240.0,-1209600.0,47900160.0,-1.8924375803183791606e9,7.47242496e10,-2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15, 1.8152105401943546773e17, -7.1661652561756670113e18};


//////////////////////////////////////////////////////////////////////////
// slow implementation
template <typename T>
static FORCEINLINE T zetaSlow(const T x, const T q) {
	
	const T precision = (T)1e-7; 									// function stops the calculation of series when next item is <= precision
		
	// if (x <= (T)1.) 
	// 	throw("zeta function: x must be > 1 !");

	// if (q <= (T)0.) 
	// 	throw("zeta function: q must be > 0 !");

	T item;
	T result = (T)0.;
// #pragma omp declare reduction (add : double,float,float16 : omp_out += omp_in) initializer(omp_priv = (T)0.)
// #pragma omp simd private(item) reduction(add:result)
	for(int i = 0; i < maxIter; ++i) {		
		
		item = math::nd4j_pow((q + i),-x);
		result += item;
		
		if(item <= precision)
			break;
	}

	return result;
}

//////////////////////////////////////////////////////////////////////////
// fast implementation, it is based on Euler-Maclaurin summation formula
    template <typename T>
    T zeta(const T x, const T q) {

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
            t = a * b / coeff[i];
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

//////////////////////////////////////////////////////////////////////////
// calculate the Hurwitz zeta function for arrays
template <typename T>
static NDArray zeta_(const NDArray& x, const NDArray& q) {

	auto result = NDArrayFactory::_create(&x, false, x.getWorkspace());

#pragma omp parallel for if(x.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)	
	for(int i = 0; i < x.lengthOf(); ++i)
		result.putScalar(i, zeta<T>(x.getScalar<T>(i), q.getScalar<T>(i)));

	return result;
}

	NDArray zeta(const NDArray& x, const NDArray& q) {
		BUILD_SINGLE_SELECTOR(x.dataType(), zeta_, (x, q), FLOAT_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template NDArray zeta_, (const NDArray& x, const NDArray& q), FLOAT_TYPES);


    template float16 zeta(float16, float16);
    template float zeta(float, float);
    template double zeta(double, double);
}
}
}

