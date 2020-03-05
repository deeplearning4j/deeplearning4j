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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <ops/declarable/helpers/activations.h>
#include <helpers/ShapeUtils.h>
#include <numeric>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ void preluCuda(const void *vx, const Nd4jLong *xShapeInfo,
		   			 	  const void *vy, const Nd4jLong *yShapeInfo,
						        void *vz) {

	const auto x = reinterpret_cast<const X*>(vx);
	const auto y = reinterpret_cast<const Y*>(vy);
		  auto z = reinterpret_cast<X*>(vz);

	__shared__ Nd4jLong xzLen;
	__shared__ int xzRank, yRank;

	if (threadIdx.x == 0) {
		xzLen = shape::length(xShapeInfo);

		xzRank = shape::rank(xShapeInfo);
		yRank  = shape::rank(yShapeInfo);
	}
	__syncthreads();

	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	Nd4jLong coords[MAX_RANK];

	for (int i = tid; i < xzLen; i += blockDim.x * gridDim.x) {
    	shape::index2coords(i, xShapeInfo, coords);

		const auto xzOffset = shape::getOffset(xShapeInfo, coords);
		const auto xVal = x[xzOffset];

		if(xVal < 0) {
			for (uint j = 0; j < yRank; ++j)
				if(yShapeInfo[j + 1] == 1)
					coords[j + 1] = 0;

			z[xzOffset] = xVal * y[shape::getOffset(yShapeInfo, coords + 1)];
		}
		else
			z[xzOffset] = xVal;
	}
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
linkage void preluCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz) {
	preluCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz);
}

///////////////////////////////////////////////////////////////////
void prelu(sd::LaunchContext * context, const NDArray& input, const NDArray& alpha, NDArray& output) {

	PointersManager manager(context, "prelu");

    const int threadsPerBlock = 256;
    const int blocksPerGrid = 512;
    const int sharedMem = 512;

	const auto xType = input.dataType();
	const auto yType = alpha.dataType();

	NDArray::prepareSpecialUse({&output}, {&input, &alpha});
	BUILD_SINGLE_SELECTOR_TWICE(xType, preluCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), output.getSpecialBuffer()), FLOAT_TYPES);
	NDArray::registerSpecialUse({&output}, {&input, &alpha});

	manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__global__ linkage void preluBPCuda(const void *vIn,    const Nd4jLong *inShapeInfo,
								   const void *vAlpha, const Nd4jLong *alphaShapeInfo,
								   const void *vdLdO,  const Nd4jLong *dLdOShapeInfo,
										 void *vdLdI,  const Nd4jLong *dLdIShapeInfo,
										 void *vdLdA,  const Nd4jLong *dLdAShapeInfo) {

	const auto in    = reinterpret_cast<const X*>(vIn);
	const auto alpha = reinterpret_cast<const Y*>(vAlpha);
	const auto dLdO  = reinterpret_cast<const Y*>(vdLdO);
		  auto dLdI  = reinterpret_cast<Y*>(vdLdI);
		  auto dLdA  = reinterpret_cast<Y*>(vdLdA);

	__shared__ Nd4jLong inLen, totalThreads;
	__shared__ int inRank, alphaRank;

	if (threadIdx.x == 0) {
		inLen = shape::length(inShapeInfo);
		totalThreads = gridDim.x * blockDim.x;

		inRank     = shape::rank(inShapeInfo);
		alphaRank  = shape::rank(alphaShapeInfo);
	}
	__syncthreads();

	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	Nd4jLong coords[MAX_RANK];

	for (int i = tid; i < inLen; i += totalThreads) {
    	shape::index2coords(i, inShapeInfo, coords);

		const auto inOffset   = shape::getOffset(inShapeInfo, coords);
		const auto dLdOOffset = shape::getOffset(dLdOShapeInfo, coords);
		const auto dLdIOffset = shape::getOffset(dLdIShapeInfo, coords);

		const auto xVal = in[inOffset];
		const auto grO  = dLdO[dLdOOffset];

		if(xVal < 0) {

			for (uint j = 0; j < alphaRank; ++j)
				if(alphaShapeInfo[j + 1] == 1)
					coords[j + 1] = 0;

			const auto alphaOffset = shape::getOffset(alphaShapeInfo, coords + 1);
			const auto dLdAOffset  = shape::getOffset(dLdAShapeInfo, coords + 1);

			dLdI[dLdIOffset] =  grO * alpha[alphaOffset];

			sd::math::atomics::nd4j_atomicAdd<Y>(&dLdA[dLdAOffset], static_cast<Y>(grO * xVal));
		}
		else
			dLdI[dLdIOffset] = grO;
	}
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Y>
__host__ linkage void preluBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vIn, const Nd4jLong *inShapeInfo, const void *vAlpha, const Nd4jLong *alphaShapeInfo, const void *vdLdO,  const Nd4jLong *dLdOShapeInfo, void *vdLdI,  const Nd4jLong *dLdIShapeInfo, void *vdLdA,  const Nd4jLong *dLdAShapeInfo) {

	preluBPCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vIn, inShapeInfo, vAlpha, alphaShapeInfo, vdLdO, dLdOShapeInfo, vdLdI, dLdIShapeInfo, vdLdA, dLdAShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
void preluBP(sd::LaunchContext* context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {
    dLdA.nullify();

	PointersManager manager(context, "preluBP");

    const int threadsPerBlock = 256;
    const int blocksPerGrid = 512;
    const int sharedMem = 512;

	const auto xType = input.dataType();
	const auto zType = alpha.dataType();

	NDArray::prepareSpecialUse({&dLdI, &dLdA}, {&input, &alpha, &dLdO});
	BUILD_SINGLE_SELECTOR_TWICE(xType, preluBPCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), dLdO.getSpecialBuffer(),  dLdO.getSpecialShapeInfo(), dLdI.getSpecialBuffer(), dLdI.getSpecialShapeInfo(), dLdA.getSpecialBuffer(), dLdA.getSpecialShapeInfo()), FLOAT_TYPES);
	NDArray::registerSpecialUse({&dLdI, &dLdA}, {&input, &alpha, &dLdO});

	manager.synchronize();
}


///////////////////////////////////////////////////////////////////
template<typename T>
__device__ void softMaxForVectorCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

	// logic of this kernel is based on assumption gridDim = 1

	const auto x = reinterpret_cast<const T*>(vx);
		  auto z = reinterpret_cast<T*>(vz);

	__shared__ Nd4jLong  len;
	__shared__ int numOfIters;
	__shared__ T* shmem;

	if (threadIdx.x == 0) {
		extern __shared__ char shared[];
		shmem = reinterpret_cast<T*>(shared);
		len = shape::length(xShapeInfo);
		numOfIters = (len + blockDim.x - 1) / blockDim.x;   // ceil (len / blockDim.x)
	}
	__syncthreads();

	T temp = -DataTypeUtils::max<T>();	// set start value to compare with at first iteration, FIXME: what if T is unsigned ??

	// ************ evaluate max element in input array x ************ //
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong xOffset = shape::getIndexOffset(elemIdx, xShapeInfo);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[xOffset] : sd::math::nd4j_max<T>(x[xOffset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = sd::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
			__syncthreads();
		}

		temp = shmem[0];	// save max value calculated at current iteration
	}

	const T max = temp;
	temp = 0;

	// ************ evaluate value of exp(x[offset] - max) per each element, store it to shared memory shmem ************ //
	// at the same evaluate sum of exponents, sum will be stored in shmem[0]
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong xOffset = shape::getIndexOffset(elemIdx, xShapeInfo);
			const Nd4jLong zOffset = shape::getIndexOffset(elemIdx, zShapeInfo);
			z[zOffset] = sd::math::nd4j_exp<T, T>(x[xOffset] - max);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? z[zOffset] : (z[zOffset] + temp); // take into account sum element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = 0;

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] += shmem[threadIdx.x + s];
			__syncthreads();
		}

		temp = shmem[0];	// save sum calculated at current iteration
	}

	// ************ evaluate z[offset] / sum  ************ //
	for (int i = 0; i < numOfIters; ++i) {
		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx >= len) continue;
		const Nd4jLong zOffset = shape::getIndexOffset(elemIdx, zShapeInfo);
		z[zOffset] /= shmem[0];
	}
}

template<typename T>
__global__ void softMaxForVectorCudaGlobal(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

	softMaxForVectorCuda<T>(vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
template <typename T>
linkage void softMaxForVectorCudaLauncher(const cudaStream_t* stream, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo) {

	softMaxForVectorCudaGlobal<T><<<1, MAX_NUM_THREADS / 4 , (MAX_NUM_THREADS / 4) * sizeof(T) + 512, *stream>>>(vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void softMaxCuda(const void* vx, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                         void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets) {

    const auto x = reinterpret_cast<const T*>(vx);
          auto z = reinterpret_cast<T*>(vz);

    const auto* xTad = x + xOffsets[blockIdx.x];
          auto* zTad = z + zOffsets[blockIdx.x];

    softMaxForVectorCuda<T>(xTad, xTadShapeInfo, zTad, zTadShapeInfo);
}

///////////////////////////////////////////////////////////////////
template<typename T>
static void softMaxCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream,
                                const void* vx, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets,
                                	  void* vz, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets) {

    softMaxCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xTadShapeInfo, xOffsets, vz, zTadShapeInfo, zOffsets);
}


//////////////////////////////////////////////////////////////////////////
void softmax(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	const int rank = input.rankOf();

	PointersManager manager(context, "helpers::softmax");

	if(input.isVector()) {

		if(rank == 1 || input.sizeAt(dimension) != 1) {
			NDArray::prepareSpecialUse({&output}, {&input});
			BUILD_SINGLE_SELECTOR(input.dataType(), softMaxForVectorCudaLauncher, (context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer(), output.getSpecialShapeInfo()), FLOAT_TYPES);
			NDArray::registerSpecialUse({&output}, {&input});
		}
		else
			output = 1.;
	}
	else {

		auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), {dimension});
        auto packZ = sd::ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), {dimension});

        const int threadsPerBlock = MAX_NUM_THREADS / 4;
        const int blocksPerGrid = packZ.numberOfTads();
        const int sharedMem = input.sizeOfT() * threadsPerBlock + 512;

        NDArray::prepareSpecialUse({&output}, {&input});
    	BUILD_SINGLE_SELECTOR(input.dataType(), softMaxCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.getSpecialBuffer(), packX.specialShapeInfo(), packX.specialOffsets(), output.specialBuffer(), packZ.specialShapeInfo(), packZ.specialOffsets()), FLOAT_TYPES);
    	NDArray::registerSpecialUse({&output}, {&input});

		// auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, {dimension}, true);
		// (input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
		// auto sumAlongDim = output.reduceAlongDimension(reduce::Sum, {dimension}, true);
		// output /= sumAlongDim;
		// input.tickReadDevice();
	}


	manager.synchronize();

	output.tickWriteDevice();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__  void logSoftMaxForVectorCuda(const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	// logic of this kernel is based on assumption gridDim = 1

	const auto x = reinterpret_cast<const T*>(vx);
		  auto z = reinterpret_cast<T*>(vz);

	__shared__ Nd4jLong  len;
	__shared__ int numOfIters;
	__shared__ T* shmem;

	if (threadIdx.x == 0) {
		extern __shared__ char shared[];
		shmem = reinterpret_cast<T*>(shared);
		len = shape::length(xzShapeInfo);
		numOfIters = (len + blockDim.x - 1) / blockDim.x;   // ceil (len / blockDim.x)
	}
	__syncthreads();

	T temp = -DataTypeUtils::max<T>();	// set start value to compare with at first iteration, FIXME: what if T is unsigned ??

	// ************ evaluate max element in input array x ************ //
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : sd::math::nd4j_max<T>(x[offset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = sd::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
			__syncthreads();
		}

		temp = shmem[0];	// save max value calculated at current iteration
	}

	const T max = temp;
	temp = 0;

	// ************ evaluate value of exp(x[offset] - max) per each element, store it to shared memory shmem ************ //
	// at the same time evaluate sum of exponents, sum will be stored in shmem[0]
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
			z[offset] = sd::math::nd4j_exp<T, T>(x[offset] - max);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? z[offset] : (z[offset] + temp); // take into account sum element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = 0;

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] += shmem[threadIdx.x + s];
			__syncthreads();
		}

		temp = shmem[0];	// save sum calculated at current iteration
	}

	// ************ evaluate log(z[offset] / sum)  ************ //
	for (int i = 0; i < numOfIters; ++i) {
		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx >= len) continue;
		const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
		z[offset] = sd::math::nd4j_log<T,T>(z[offset] / shmem[0]);
	}
}

///////////////////////////////////////////////////////////////////
template <typename T>
linkage void logSoftMaxForVectorCudaLauncher(const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	logSoftMaxForVectorCuda<T><<<1, MAX_NUM_THREADS, MAX_NUM_THREADS * sizeof(T) + 512, *stream>>>(vx, xzShapeInfo, vz);
}

//////////////////////////////////////////////////////////////////////////
void logSoftmax(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	const int rank = input.rankOf();

	if(input.isVector()) {

		if(rank == 1 || input.sizeAt(dimension) != 1) {
			BUILD_SINGLE_SELECTOR(input.dataType(), logSoftMaxForVectorCudaLauncher, (context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer()), FLOAT_TYPES);
			input.tickReadDevice();
		}
		else
			output = 0.;
	}
	else {

		auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, {dimension}, true);
		(input - maxAlongDim).applyTransform(transform::Exp, output); // output contains exponents temporarily
		auto sumAlongDim = output.reduceAlongDimension(reduce::Sum, {dimension}, true);
		output /= sumAlongDim;
		output.applyTransform(transform::Log, output);
		input.tickReadDevice();
	}

	PointersManager manager(context, "helpers::logSoftmax");
	manager.synchronize();

	output.tickWriteDevice();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ linkage void softMaxDerivForVectorCuda(const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	// logic of this kernel is based on assumption gridDim = 1

	const auto x = reinterpret_cast<const T*>(vx);
		  auto z = reinterpret_cast<T*>(vz);

	__shared__ Nd4jLong  len;
	__shared__ int numOfIters;
	__shared__ T* shmem;

	if (threadIdx.x == 0) {
		extern __shared__ char shared[];
		shmem = reinterpret_cast<T*>(shared);
		len = shape::length(xzShapeInfo);
		numOfIters = (len + blockDim.x - 1) / blockDim.x;   // ceil (len / blockDim.x)
	}
	__syncthreads();

	T temp = -DataTypeUtils::max<T>();	// set start value to compare with at first iteration, FIXME: what if T is unsigned ??

	// ************ evaluate max element in input array x ************ //
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : sd::math::nd4j_max<T>(x[offset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = sd::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
			__syncthreads();
		}

		temp = shmem[0];	// save max value calculated at current iteration
	}

	const T max = temp;
	temp = 0;

	// ************ evaluate value of exp(x[offset] - max) per each element, store it to shared memory shmem ************ //
	// at the same evaluate sum of exponents, sum will be stored in shmem[0]
	for (int i = 0; i < numOfIters; ++i) {

		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx < len) {
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
			z[offset] = sd::math::nd4j_exp<T, T>(x[offset] - max);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? z[offset] : (z[offset] + temp); // take into account sum element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = 0;

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] += shmem[threadIdx.x + s];
			__syncthreads();
		}

		temp = shmem[0];	// save sum calculated at current iteration
	}

	// ************ evaluate (z[offset] / sum) and derivative z[offset] = z[offset] * (1 - z[offset]) ************ //
	for (int i = 0; i < numOfIters; ++i) {
		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx >= len) continue;
		const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo);
		z[offset] /= shmem[0];
		z[offset] *= (1.f - z[offset]);		// derivative
	}
}

///////////////////////////////////////////////////////////////////
template <typename T>
linkage void softMaxDerivForVectorCudaLauncher(const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	softMaxDerivForVectorCuda<T><<<1, MAX_NUM_THREADS, MAX_NUM_THREADS * sizeof(T) + 512, *stream>>>(vx, xzShapeInfo, vz);
}

///////////////////////////////////////////////////////////////////
void softmaxDerivative(sd::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	const int rank = input.rankOf();
	int temp;

	if(shape::isCommonVector(input.getShapeInfo(), temp)) {

		BUILD_SINGLE_SELECTOR(input.dataType(), softMaxDerivForVectorCudaLauncher, (context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer()), FLOAT_TYPES);
		input.tickReadDevice();
	}
	else {

		auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDimension(reduce::Max, {dimension}, true);
		(input - maxAlongDim).applyTransform(transform::Exp, output); // output contains exponents temporarily
		auto sumAlongDim = output.reduceAlongDimension(reduce::Sum, {dimension}, true);
		output /= sumAlongDim;
		output *= (1.f - output);	// derivative
		input.tickReadDevice();
	}

	PointersManager manager(context, "helpers::softmaxDerivative");
	manager.synchronize();

	output.tickWriteDevice();
}


	template <typename T>
	linkage void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
		auto routine = LAMBDA_T(_x, threshold) {
			return _x > (T)threshold ? _x: (T)0.f;
		};
		const_cast<NDArray&>(input).applyLambda(routine, output);
	}

	void thresholdRelu(sd::LaunchContext * context, NDArray const& input, double threshold, NDArray& output) {
		BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
	}

	template <typename T>
	linkage void thresholdReluDerivative_(NDArray* input, double theta, NDArray* dLdO, NDArray* output) {
        auto derivative = LAMBDA_TT(_x, grO, theta) {if (_x > theta) return grO; else return static_cast<T>(0); };

        input->applyPairwiseLambda(*dLdO, derivative, *output);
	}

	void thresholdReluDerivative(sd::LaunchContext * context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
		BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdO, output), FLOAT_TYPES);
	}

}
}
}

