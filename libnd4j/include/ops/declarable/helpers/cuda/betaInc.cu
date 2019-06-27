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
#include <PointersManager.h>

namespace nd4j {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
// modified Lentz’s algorithm for continued fractions,
// reference: Lentz, W.J. 1976, “Generating Bessel Functions in Mie Scattering Calculations Using Continued Fractions,”
template <typename T>
__device__ T continuedFractionCuda(const T a, const T b, const T x) {

	extern __shared__ unsigned char shmem[];
	T* coeffs = reinterpret_cast<T*>(shmem);

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
		// d
		d = static_cast<T>(1) + coeffs[i - 1] * d;
		if(math::nd4j_abs<T>(d) < min)
			d = min;
		d = static_cast<T>(1) / d;
		// c
		c = static_cast<T>(1) + coeffs[i - 1] / c;
		if(math::nd4j_abs<T>(c) < min)
			c = min;
		// f
		f *= c * d;


		/***** odd part *****/
		// d
		d = static_cast<T>(1) + coeffs[i] * d;
		if(math::nd4j_abs<T>(d) < min)
			d = min;
		d = static_cast<T>(1) / d;
		// c
		c = static_cast<T>(1) + coeffs[i] / c;
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
__device__ T betaIncCoreCuda(T a, T b, T x) {

	const T gammaPart = lgamma(a) + lgamma(b) - lgamma(a + b);
    const T front = math::nd4j_exp<T,T>(math::nd4j_log<T, T>(x) * a + math::nd4j_log<T, T>(1 - x) * b - gammaPart) / a;

	if (x <= (a + static_cast<T>(1)) / (a + b + static_cast<T>(2)))
		return front * continuedFractionCuda(a, b, x);
	else  // symmetry relation
		return static_cast<T>(1) - front * continuedFractionCuda(b, a, static_cast<T>(1) - x);
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ void betaIncForArrayCuda(const void* va, const Nd4jLong* aShapeInfo,
									const void* vb, const Nd4jLong* bShapeInfo,
									const void* vx, const Nd4jLong* xShapeInfo,
										  void* vz, const Nd4jLong* zShapeInfo) {

    extern __shared__ unsigned char shmem[];
    T* sharedMem = reinterpret_cast<T*>(shmem);

    const Nd4jLong j = blockIdx.x;			// one block per each element

    Nd4jLong len = shape::length(xShapeInfo);

    const T  a = *(reinterpret_cast<const T*>(va) + shape::getIndexOffset(j, aShapeInfo, len));
    const T  b = *(reinterpret_cast<const T*>(vb) + shape::getIndexOffset(j, bShapeInfo, len));
    const T  x = *(reinterpret_cast<const T*>(vx) + shape::getIndexOffset(j, xShapeInfo, len));
    	  T& z = *(reinterpret_cast<T*>(vz) 	 	 + shape::getIndexOffset(j, zShapeInfo, len));

    // t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
   	if(a == b && x == static_cast<T>(0.5)) {
		z = static_cast<T>(0.5);
		return;
   	}

	if (x == static_cast<T>(0) || x == static_cast<T>(1)) {
		z = x;
		return;
	}

   	if(threadIdx.x % 2 == 0) { 	/***** even part *****/
		const int m = threadIdx.x + 1;
		sharedMem[threadIdx.x] = m * (b - m) * x / ((a + 2 * m - static_cast<T>(1)) * (a + 2 * m));
	}
	else {						/***** odd part *****/
		const int m = threadIdx.x;
		sharedMem[threadIdx.x] = -(a + m) * (a + b + m) * x / ((a + 2 * m + static_cast<T>(1)) * (a + 2 * m));
	}

	__syncthreads();

	if(threadIdx.x == 0)
		z = betaIncCoreCuda(a, b, x);
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void betaIncForArrayCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                		const void* va, const Nd4jLong* aShapeInfo,
										const void* vb, const Nd4jLong* bShapeInfo,
										const void* vx, const Nd4jLong* xShapeInfo,
									  		  void* vz, const Nd4jLong* zShapeInfo) {

    betaIncForArrayCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(va, aShapeInfo, vb, bShapeInfo, vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
// overload betaInc for arrays, shapes of a, b and x must be the same !!!
void betaInc(nd4j::LaunchContext* context, const NDArray& a, const NDArray& b, const NDArray& x, NDArray& output) {

    const int threadsPerBlock = maxIter;
    const int blocksPerGrid = output.lengthOf();
    const int sharedMem = output.sizeOfT() * threadsPerBlock  + 128;

    const auto xType = x.dataType();

    PointersManager manager(context, "betaInc");

    NDArray::prepareSpecialUse({&output}, {&a, &b, &x});
	BUILD_SINGLE_SELECTOR(xType, betaIncForArrayCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), a.getSpecialBuffer(), a.getSpecialShapeInfo(), b.getSpecialBuffer(), b.getSpecialShapeInfo(), x.getSpecialBuffer(), x.getSpecialShapeInfo(), output.specialBuffer(), output.specialShapeInfo()), FLOAT_TYPES);
	NDArray::registerSpecialUse({&output}, {&a, &b, &x});

    manager.synchronize();
}


}
}
}

