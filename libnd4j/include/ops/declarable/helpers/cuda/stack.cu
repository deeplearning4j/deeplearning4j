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
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

//	Nd4jLong* tadOnlyInputShapeInfo,  Nd4jLong *tadInputOffsets,
//	Nd4jLong* tadOnlyOutputShapeInfo, Nd4jLong *tadOutputOffsets

//	template <typename T>
//	static __global__ void stackKernelScalar(void const* inputList[], void* outputBuffer, Nd4jLong* outputShape, Nd4jLong outputLength) {
//		auto tid = blockIdx.x * blockDim.x + threadIdx.x;
//		int totalThreads = gridDim.x * blockDim.x;
//		//const auto resultLength = shape::length(outputShape);
//		for (Nd4jLong i = tid; i < outputLength; i += totalThreads) {
//			//auto yOffset = shape::subArrayOffset(i, outputShape, inputShape);
//			//printf(">> %lld\n", i);
//			auto xOffset = shape::getIndexOffset(i, outputShape, outputLength);
//			printf(">> %lld\n", xOffset);
//
//			//*(reinterpret_cast<T *>(outputBuffer) + xOffset) = *(reinterpret_cast<T const *>(inputList[xOffset]));
//		}
//	}

	template <typename T>
	static __global__ void stackKernel(void* inputList[], void* inputShapeList[], size_t inputListLength, Nd4jLong arrLen, void* outputBuffer, Nd4jLong* outputShape) {

		__shared__ int arrIdx, blocksPerArr;
		__shared__ T *x, *z;
		__shared__ Nd4jLong *zShapeInfo, *xShapeInfo, arrLenPerBlock, start, end;

		if (threadIdx.x == 0) {

			blocksPerArr = (gridDim.x + inputListLength - 1) / inputListLength;     // ceil
			arrIdx = blockIdx.x / blocksPerArr;

			x = reinterpret_cast<T*>(inputList[arrIdx]);
			z = reinterpret_cast<T*>(outputBuffer);
			xShapeInfo = reinterpret_cast<Nd4jLong*>(inputShapeList[arrIdx]);
			zShapeInfo = reinterpret_cast<Nd4jLong*>(outputShape);
			//arrLen = shape::length(xShapeInfo);

			arrLenPerBlock = (arrLen + blocksPerArr - 1) / blocksPerArr;  // ceil

			start = (blockIdx.x % blocksPerArr) * arrLenPerBlock;
			end   = (start + arrLenPerBlock) > arrLen ? arrLen : (start + arrLenPerBlock);
			printf("Block: [%i]; arrIdx: [%i]; start: [%i]; end: [%i], arrLen: [%i], arrLenPerBlock: [%i]\n", blockIdx.x, arrIdx, start, end, arrLen, arrLenPerBlock);
		}

		__syncthreads();
        //for (Nd4jLong arr = blockIdx.x; arr < inputListLength; arr += gridDim.x) {
		for (Nd4jLong i = start + threadIdx.x; i < end; i += blockDim.x)
				z[shape::getIndexOrderOffset(i, zShapeInfo, arrLen,
											 shape::order(zShapeInfo))] = x[shape::getIndexOrderOffset(i, xShapeInfo,
																									   arrLen,
																									   shape::order(
																											   xShapeInfo))];

	}
	///////////////////////////////////////////////////////////////////
	template <typename T>
	static void stack_(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim) {
		if(inArrs[0]->isScalar()) {
            outArr->lazyAllocateBuffer();

//#pragma omp parallel for
			for (size_t i = 0; i < inArrs.size(); ++i) {
                inArrs[i]->syncToHost();

                outArr->p(i, inArrs[i]->e<T>(0));
            }
			outArr->syncToDevice();
		}
		else {
			Nd4jLong **dInShapeInfo;
			void **dInBuffers;
			std::vector<void const*> inputList(inArrs.size());
			std::vector<Nd4jLong const*> inputShapeList(inArrs.size());
			auto stream = context->getCudaStream();

			for (size_t i = 0; i < inputList.size(); ++i) {
				inputList[i] = inArrs[i]->getSpecialBuffer();
				inputShapeList[i] = inArrs[i]->getSpecialShapeInfo();
			}

			cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void **>(&dInBuffers), inputList.size() * sizeof(void*));
			if(cudaResult != 0) throw cuda_exception::build("helpers::stack_: cannot allocate global memory on device", cudaResult);
			cudaResult = cudaMalloc(reinterpret_cast<void **>(&dInShapeInfo), inputShapeList.size() * sizeof(Nd4jLong*));
			if(cudaResult != 0) throw cuda_exception::build("helpers::stack_: cannot allocate global memory on device", cudaResult);

			cudaMemcpyAsync(dInBuffers,    inputList.data(),    inputList.size()  * sizeof(void*),       cudaMemcpyHostToDevice, *stream);
			cudaMemcpyAsync(dInShapeInfo,  inputShapeList.data(),  inputShapeList.size() * sizeof(Nd4jLong*),  cudaMemcpyHostToDevice, *stream);

            dim3 launchDims(256, 512, 8192);

			stackKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>((void**)dInBuffers, (void**)dInShapeInfo, inputList.size(), inArrs[0]->lengthOf(), outArr->specialBuffer(), outArr->specialShapeInfo());

			cudaResult = cudaFree(dInBuffers);
			if(cudaResult != 0)
				throw cuda_exception::build("helpers::stack_: cannot deallocate global memory on device for buffer list", cudaResult);
			cudaResult = cudaFree(dInShapeInfo);
			if(cudaResult != 0)
				throw cuda_exception::build("helpers::stack_: cannot deallocate global memory on device for shape list", cudaResult);

		}
	}

	void stack(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr->dataType(), stack_, (context, inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray* outArr, const int dim), LIBND4J_TYPES);

}
}
}

