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

#include <op_boilerplate.h>
#include <ops/declarable/helpers/activations.h>
#include <ShapeUtils.h>
#include <numeric>
#include <PointersManager.h>

namespace nd4j    {
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

	__shared__ Nd4jLong  len;

	if (threadIdx.x == 0)
		len = shape::length(xShapeInfo);

	__syncthreads();

	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	const auto totalThreads = gridDim.x * blockDim.x;

	for (int i = tid; i < len; i += totalThreads) {

		const auto xzOffset = shape::getIndexOffset(i, xShapeInfo, len);
		const auto xVal     = x[xzOffset];

		if(xVal < 0)
			z[xzOffset] = xVal * y[shape::subArrayOffset(i, xShapeInfo, yShapeInfo)];
		else
			z[xzOffset] = xVal;
	}
}

///////////////////////////////////////////////////////////////////
template<typename X, typename Y>
linkage void preluCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz) {

	preluCuda<X, Y><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz);
}

///////////////////////////////////////////////////////////////////
void prelu(nd4j::LaunchContext * context, const NDArray& input, const NDArray& alpha, NDArray& output) {
	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	if(!alpha.isActualOnDeviceSide()) alpha.syncToDevice();

	const auto xType = input.dataType();
	const auto yType = alpha.dataType();
	int threadsPerBlock = MAX_NUM_THREADS;
	int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

	BUILD_DOUBLE_SELECTOR(xType, yType, preluCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), output.getSpecialBuffer()), LIBND4J_TYPES, FLOAT_TYPES);

	input.tickReadHost();
	alpha.tickReadHost();
	output.tickWriteDevice();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ void softMaxForVectorCuda(const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : nd4j::math::nd4j_max<T>(x[offset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = nd4j::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			z[offset] = nd4j::math::nd4j_exp<T, T>(x[offset] - max);
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

	// ************ evaluate z[offset] / sum  ************ //
	for (int i = 0; i < numOfIters; ++i) {
		const Nd4jLong elemIdx = i * blockDim.x + threadIdx.x;
		if(elemIdx >= len) continue;
		const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
		z[offset] /= shmem[0];
	}
}

///////////////////////////////////////////////////////////////////
template <typename T>
linkage void softMaxForVectorCudaLauncher(const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	softMaxForVectorCuda<T><<<1, MAX_NUM_THREADS, MAX_NUM_THREADS * sizeof(T) + 512, *stream>>>(vx, xzShapeInfo, vz);
}

//////////////////////////////////////////////////////////////////////////
void softmax(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	const int rank = input.rankOf();

	if(input.isVector()) {

		if(rank == 1 || input.sizeAt(dimension) != 1) {
			BUILD_SINGLE_SELECTOR(input.dataType(), softMaxForVectorCudaLauncher, (context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer()), FLOAT_TYPES);
			input.tickReadDevice();
		}
		else
			output = 1.;
	}
	else {

		auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
		(input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
		auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);
		output /= sumAlongDim;
		input.tickReadDevice();
	}

	PointersManager manager(context, "helpers::softmax");
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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : nd4j::math::nd4j_max<T>(x[offset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = nd4j::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			z[offset] = nd4j::math::nd4j_exp<T, T>(x[offset] - max);
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
		const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
		z[offset] = nd4j::math::nd4j_log<T,T>(z[offset] / shmem[0]);
	}
}

///////////////////////////////////////////////////////////////////
template <typename T>
linkage void logSoftMaxForVectorCudaLauncher(const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz) {

	logSoftMaxForVectorCuda<T><<<1, MAX_NUM_THREADS, MAX_NUM_THREADS * sizeof(T) + 512, *stream>>>(vx, xzShapeInfo, vz);
}

//////////////////////////////////////////////////////////////////////////
void logSoftmax(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

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

		auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
		(input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
		auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);
		output /= sumAlongDim;
		output.applyTransform(transform::Log);
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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : nd4j::math::nd4j_max<T>(x[offset], temp);	// take into account max element evaluated on previous iteration and stored in temp
		}
		else
			shmem[threadIdx.x] = -DataTypeUtils::max<T>();	// FIXME: what if T is unsigned ??

		__syncthreads();

		for (int s = blockDim.x / 2; s > 0; s /= 2) {
			if(threadIdx.x < s)
				shmem[threadIdx.x] = nd4j::math::nd4j_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
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
			const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
			z[offset] = nd4j::math::nd4j_exp<T, T>(x[offset] - max);
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
		const Nd4jLong offset = shape::getIndexOffset(elemIdx, xzShapeInfo, len);
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
void softmaxDerivative(nd4j::LaunchContext * context, const NDArray& input, NDArray& output, const int dimension) {

	if(!input.isActualOnDeviceSide()) input.syncToDevice();
	const int rank = input.rankOf();
	int temp;

	if(shape::isCommonVector(input.getShapeInfo(), temp)) {

		BUILD_SINGLE_SELECTOR(input.dataType(), softMaxDerivForVectorCudaLauncher, (context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), output.getSpecialBuffer()), FLOAT_TYPES);
		input.tickReadDevice();
	}
	else {

		auto maxAlongDim = const_cast<NDArray&>(input).reduceAlongDims(reduce::Max, {dimension}, true);
		(input - maxAlongDim).applyTransform(transform::Exp, &output); // output contains exponents temporarily
		auto sumAlongDim = output.reduceAlongDims(reduce::Sum, {dimension}, true);
		output /= sumAlongDim;
		output *= (1.f - output);	// derivative
		input.tickReadDevice();
	}

	PointersManager manager(context, "helpers::softmaxDerivative");
	manager.synchronize();

	output.tickWriteDevice();
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

	__shared__ Nd4jLong alphaLen;

	if (threadIdx.x == 0)
		alphaLen = shape::length(alphaShapeInfo);

	__syncthreads();

	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= alphaLen) return;

	Nd4jLong inputIdxs[MAX_RANK*2];
	int numIdxs = shape::outerArrayOffsets(inputIdxs, i, inShapeInfo, alphaShapeInfo);
	Nd4jLong dLdOIdxs[MAX_RANK*2];
	shape::outerArrayOffsets(dLdOIdxs, i, dLdOShapeInfo, alphaShapeInfo);
	Nd4jLong dLdIIdxs[MAX_RANK*2];
	shape::outerArrayOffsets(dLdIIdxs, i, dLdIShapeInfo, alphaShapeInfo);

	const auto alphaOffset = shape::getIndexOffset(i, alphaShapeInfo, alphaLen);
	const auto dLdAOffset  = shape::getIndexOffset(i, dLdAShapeInfo, alphaLen);

	for(Nd4jLong j = 0; j < numIdxs; ++j) {

		const auto inInd   = inputIdxs[j];
		const auto dLdOInd = dLdOIdxs[j];
		const auto dLdIInd = dLdIIdxs[j];

		if(in[inInd] < 0) {
			dLdI[dLdIInd] = dLdO[dLdOInd] * alpha[alphaOffset];
			auto prevVal = dLdA[dLdAOffset];
			prevVal = prevVal + dLdO[dLdOInd] * in[inInd];
			dLdA[dLdAOffset] = prevVal;
		}
		else
			dLdI[dLdIInd] = dLdO[dLdOInd];
	}
}


template<typename X, typename Y>
__host__ linkage void preluBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vIn, const Nd4jLong *inShapeInfo, const void *vAlpha, const Nd4jLong *alphaShapeInfo, const void *vdLdO,  const Nd4jLong *dLdOShapeInfo, void *vdLdI,  const Nd4jLong *dLdIShapeInfo, void *vdLdA,  const Nd4jLong *dLdAShapeInfo) {

	preluBPCuda<X, Y><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(vIn, inShapeInfo, vAlpha, alphaShapeInfo, vdLdO, dLdOShapeInfo, vdLdI, dLdIShapeInfo, vdLdA, dLdAShapeInfo);
}


	//////////////////////////////////////////////////////////////////////////
	void preluBP(nd4j::LaunchContext * context, const NDArray& input, const NDArray& alpha, const NDArray& dLdO, NDArray& dLdI, NDArray& dLdA) {

		if(!input.isActualOnDeviceSide()) input.syncToDevice();
		if(!alpha.isActualOnDeviceSide()) alpha.syncToDevice();
		if(!dLdO.isActualOnDeviceSide())  dLdO.syncToDevice();

		const auto xType = input.dataType();
		const auto zType = dLdO.dataType();

		int threadsPerBlock = MAX_NUM_THREADS;
		int blocksPerGrid = (alpha.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

		BUILD_DOUBLE_SELECTOR(xType, zType, preluBPCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.getSpecialBuffer(), input.getSpecialShapeInfo(), alpha.getSpecialBuffer(), alpha.getSpecialShapeInfo(), dLdO.getSpecialBuffer(),  dLdO.getSpecialShapeInfo(), dLdI.getSpecialBuffer(), dLdI.getSpecialShapeInfo(), dLdA.getSpecialBuffer(), dLdA.getSpecialShapeInfo()), LIBND4J_TYPES, FLOAT_TYPES);

		input.tickReadHost();
		alpha.tickReadHost();
		dLdO.tickReadHost();
		dLdI.tickWriteDevice();
		dLdA.tickWriteDevice();

	}


	template <typename T>
	linkage void thresholdRelu_(NDArray const& input, double threshold, NDArray& output) {
		auto routine = LAMBDA_T(_x, threshold) {
			return _x > (T)threshold ? _x: (T)0.f;
		};
		const_cast<NDArray&>(input).applyLambda(routine, &output);
	}

	void thresholdRelu(nd4j::LaunchContext * context, NDArray const& input, double threshold, NDArray& output) {
		BUILD_SINGLE_SELECTOR(input.dataType(), thresholdRelu_, (input, threshold, output), FLOAT_TYPES);
	}

	template <typename T>
	linkage void thresholdReluDerivative_(NDArray* input, double theta, NDArray* dLdO, NDArray* output) {

	}

	void thresholdReluDerivative(nd4j::LaunchContext * context, NDArray* input, double threshold, NDArray* dLdO, NDArray* output) {
		BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdO, output), FLOAT_TYPES);
	}


BUILD_SINGLE_TEMPLATE(template void thresholdReluDerivative_, (NDArray* input, double threshold, NDArray* dLdO, NDArray* output), FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void preluCudaLauncher,   (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_DOUBLE_TEMPLATE(template void preluBPCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void *vIn, const Nd4jLong *inShapeInfo, const void *vAlpha, const Nd4jLong *alphaShapeInfo, const void *vdLdO,  const Nd4jLong *dLdOShapeInfo, void *vdLdI,  const Nd4jLong *dLdIShapeInfo, void *vdLdA,  const Nd4jLong *dLdAShapeInfo), LIBND4J_TYPES, FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void softMaxForVectorCudaLauncher, (const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void softMaxDerivForVectorCudaLauncher, (const cudaStream_t* stream, const void *vx, const Nd4jLong *xzShapeInfo, void *vz), FLOAT_TYPES);


}
}
}

