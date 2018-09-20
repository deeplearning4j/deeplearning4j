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
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_
#ifdef _OPENMP
#include <omp.h>
#endif
#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/shape.h>
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif


#include "legacy_ops.h"

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

/**
 * Transforms involving 2 arrays
 */
        template<typename X, typename Y>
        class PairWiseTransform {
        public:

#ifdef __CUDACC__

            static __host__ void execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, Nd4jLong xStride, T *y, Nd4jLong yStride, T *result, Nd4jLong resultStride, T *extraParams, Nd4jLong n);

            static __host__ void execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, Nd4jLong *xShapeInfo, T *y, Nd4jLong *yShapeInfo, T *result, Nd4jLong *resultShapeInfo, T *extraParams);

            static __device__ void transformCuda(const int opNum, Nd4jLong n, T *dx, T *y, Nd4jLong incx, Nd4jLong incy, T *extraParams, T *result, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


            template<typename OpType>
	        static __device__ void transformCuda(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transformCuda(Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transform(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes,  int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


#endif
        public:
			static void exec(
				const int opNum,
				void *dx,
				Nd4jLong *xShapeBuffer,
				void *y,
				Nd4jLong *yShapeBuffer,
				void *result,
				Nd4jLong *resultShapeBuffer,
				void *extraParams,
				Nd4jLong *indexes,
				Nd4jLong *yIndexes,
				Nd4jLong *resultIndexes);

			static void exec(
				const int opNum,
				void *dx,
				Nd4jLong *xShapeBuffer,
				void *y,
				Nd4jLong *yShapeBuffer,
				void *result,
				Nd4jLong *resultShapeBuffer,
				void *extraParams);
			
			static void exec(
				const int opNum,
				void *dx,
				Nd4jLong xStride,
				void *y,
				Nd4jLong yStride,
				void *result,
				Nd4jLong resultStride,
				void *extraParams,
				Nd4jLong n);

			template<typename OpType>
			static void exec(
                    void *vx,
                    Nd4jLong* xShapeBuffer,
                    void *vy,
                    Nd4jLong* yShapeBuffer,
                    void *vresult,
                    Nd4jLong* resultShapeBuffer,
                    void *vextraParams,
                    Nd4jLong *indexes,
                    Nd4jLong *yIndexes,
                    Nd4jLong *resultIndexes);

			template<typename OpType>
			static void exec(
                    void *vx,
                    Nd4jLong* xShapeBuffer,
                    void *vy,
                    Nd4jLong* yShapeBuffer,
                    void *vresult,
                    Nd4jLong* resultShapeBuffer,
                    void *vextraParams);

            template<typename OpType>
            static void exec(void *vx,
                             Nd4jLong xStride,
                             void *vy,
                             Nd4jLong yStride,
                             void *vresult,
                             Nd4jLong resultStride,
                             void *vextraParams,
                             const Nd4jLong n);
        };
    }
}

#endif /* PAIRWISE_TRANSFORM_H_ */
