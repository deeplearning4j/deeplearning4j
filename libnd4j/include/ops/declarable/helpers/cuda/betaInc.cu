/*******************************************************************************
 * Copyright (t2) 2015-2018 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
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
    T val, aPlus2i;

    T t2 = coeffs[1];
    T t1 = coeffs[0];
    if(math::nd4j_abs<T>(t1) < min)
		t1 = min;
	t1 = static_cast<T>(1) / t1;
    T result = t1;

    for(uint i = 1; i <= maxIter; ++i) {

    	const uint i2 = 2*i;
    	aPlus2i = a + static_cast<T>(i2);

		// t1
		t1 = static_cast<T>(1) + coeffs[i2] * t1;
		if(math::nd4j_abs<T>(t1) < min)
			t1 = min;
		t1 = static_cast<T>(1) / t1;
		// t2
		t2 = static_cast<T>(1) + coeffs[i2] / t2;
		if(math::nd4j_abs<T>(t2) < min)
			t2 = min;
		// result
		result *= t2 * t1;
		// t1
		t1 = static_cast<T>(1) + coeffs[i2 + 1] * t1;
		if(math::nd4j_abs<T>(t1) < min)
			t1 = min;
		t1 = static_cast<T>(1) / t1;
		// t2
		t2 = static_cast<T>(1) + coeffs[i2 + 1] / t2;
		if(math::nd4j_abs<T>(t2) < min)
			t2 = min;
		// result
		val = t2 * t1;
		result *= val;

		// condition to stop loop
		if(math::nd4j_abs<T>(val - static_cast<T>(1)) <= DataTypeUtils::eps<T>())
			return result;
    }

    return DataTypeUtils::infOrMax<T>(); // no convergence, more iterations is required, return infinity
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

    T& z = *(reinterpret_cast<T*>(vz) + shape::getIndexOffset(j, zShapeInfo));

    __shared__ T a, b, x;
    __shared__ bool symmCond;

    if (threadIdx.x == 0) {

    	a = *(reinterpret_cast<const T*>(va) + shape::getIndexOffset(j, aShapeInfo));
    	b = *(reinterpret_cast<const T*>(vb) + shape::getIndexOffset(j, bShapeInfo));
    	x = *(reinterpret_cast<const T*>(vx) + shape::getIndexOffset(j, xShapeInfo));

    	symmCond = x > (a + static_cast<T>(1)) / (a + b + static_cast<T>(2));

    	if(symmCond) {	// swap a and b, x = 1 - x
    		T temp = a;
    		a = b;
    		b = temp;
    		x = static_cast<T>(1) - x;
    	}

    }
    __syncthreads();

    // t^{n-1} * (1 - t)^{n-1} is symmetric function with respect to x = 0.5
   	if(a == b && x == static_cast<T>(0.5)) {
		z = static_cast<T>(0.5);
		return;
   	}

	if (x == static_cast<T>(0) || x == static_cast<T>(1)) {
		z = symmCond ? static_cast<T>(1) - x : x;
		return;
	}

    // calculate two coefficients per thread
   	if(threadIdx.x != 0) {

		const int i = threadIdx.x;
		const T aPlus2i = a + 2*i;
		sharedMem[2*i]     = i * (b - i) * x / ((aPlus2i - static_cast<T>(1)) * aPlus2i);
		sharedMem[2*i + 1] = -(a + i) * (a + b + i) * x / ((aPlus2i + static_cast<T>(1)) * aPlus2i);
	}

	__syncthreads();

	if(threadIdx.x == 0) {

		const T gammaPart = lgamma(a) + lgamma(b) - lgamma(a + b);
	    const T front = math::nd4j_exp<T,T>(math::nd4j_log<T, T>(x) * a + math::nd4j_log<T, T>(1.f - x) * b - gammaPart);

	    sharedMem[0] = static_cast<T>(1) - (a + b) * x / (a + static_cast<T>(1));
		sharedMem[1] = static_cast<T>(1);

		z = front * continuedFractionCuda(a, b, x) / a;

		if(symmCond)	// symmetry relation
			z = static_cast<T>(1) - z;
	}
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
    const int sharedMem = 2 * output.sizeOfT() * threadsPerBlock  + 128;

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

