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
 * transform.h
 *
 *  Created on: Dec 28, 2015
 *  @author: agibsonccc
 *  @author: raver119@gmail.com
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_
#include <vector>
#include <templatemath.h>
#include <ops/ops.h>
#include <ops/special_ops.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>

//#include <loops/reduce.h>
//#include <loops/scalar.h>
//#include <loops/indexreduce.h>
//#include <loops/broadcasting.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"


namespace functions {
    namespace transform {

        template<typename X, typename Z>
        class TransformFloat {
        public:

#ifdef __CUDACC__

	template<typename OpType>
	static  __device__ void transformCuda(
			T *dy,
			Nd4jLong *shapeInfo,
			T *params,
			T *result,
			Nd4jLong *resultShapeInfo,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

	template<typename OpType>
	static  __device__ void transformCuda(
			Nd4jLong n,
			T *dy,
			Nd4jLong incy,
			T *params,
			T *result,
			Nd4jLong resultStride,
			int *allocationPointer, T *reductionPointer, UnifiedSharedMemory *manager);

	static  __device__ void transformCuda(
			const int opNum,
			T *dy,
			Nd4jLong *shapeInfo,
			T *params,
			T *result,
			Nd4jLong *resultShapeInfo,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);


	static  __device__ void transformCuda(
			const int opNum,
			Nd4jLong n,
			T *dy,
			Nd4jLong incy,
			T *params,
			T *result,
			Nd4jLong resultStride,
			int *allocationPointer,
			T *reductionPointer,
			UnifiedSharedMemory *manager);

	static _CUDA_H void executeTransformStrided(dim3 launchDims, cudaStream_t *stream, int opNum, Nd4jLong n, T *x, Nd4jLong xStride, T *extraParams, T *z, Nd4jLong zStride, int *allocationPointer, T *reductionPointer);
	
	static _CUDA_H void executeTransformShaped(dim3 launchDims, cudaStream_t *stream, int opNum, T *x, Nd4jLong *xShape, int xRank, T *extraParams, T *z, Nd4jLong *zShape, int zRank, int *allocationPointer, T *reductionPointer,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

#endif


			static void exec(int opNum, void *dx, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *extraParams, Nd4jLong n);

			static void exec(int opNum, void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

			template<typename OpType>
			static ND4J_EXPORT void exec(void *dx, Nd4jLong *xShapeInfo, void *result, Nd4jLong *resultShapeInfo, void *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

			template <typename OpType>
			static ND4J_EXPORT void exec(void *dx, Nd4jLong xStride, void *result, Nd4jLong resultStride, void *extraParams, Nd4jLong n);
        };
    }
}


#endif /* TRANSFORM_H_ */
