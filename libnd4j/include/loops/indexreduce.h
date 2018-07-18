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

/*
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXREDUCE_H_
#define INDEXREDUCE_H_
#include "../helpers/shape.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dll.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include <helpers/TAD.h>


#include "../pairwise_util.h"

#include "legacy_ops.h"

namespace functions {
	namespace indexreduce {

		template<typename T>
		class IndexReduce {
		public:
#ifdef __CUDACC__

		static __device__ void transform(const int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension,int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);

	/**
	 *
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
    template<typename OpType>
	static __device__ void aggregatePartials(IndexValue<T> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements,T *extraParams);

	/**
	 * @param n n is the number of
	 *        elements to loop through
	 * @param dx the data to operate on
	 * @param xVectorInfo the meta data for the vector:
	 *                              0 is the offset
	 *                              1 is the increment/stride
	 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
	 *                              3 is the element wise stride for the buffer
	 *                              4 is the number of elements it takes to get to the next row/column/tensor
	 * @param gpuInformation
	 *                              0 is the block size
	 *                              1 is the grid size
	 *                              2 is the shared memory size
	 * @param problemDefinition
	 *                          0 is the number of elements per vector
	 *                          1 is the number of vectors
	 */
    template<typename OpType>
	static __device__ void transform(T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);


    static _CUDA_H void executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream, const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);

    static _CUDA_H void executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets);
#endif
		static T execScalar(const int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams);

		static void exec(const int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);

		template<typename OpType>
		static _CUDA_H T execScalar(T *x, Nd4jLong *xShapeInfo, T *extraParams);

		template<typename OpType>
		static _CUDA_H void exec(T *x, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfoBuffer, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffset);
		};
	}
}


#ifdef __CUDACC__



#endif

#endif /* INDEXREDUCE_H_ */

