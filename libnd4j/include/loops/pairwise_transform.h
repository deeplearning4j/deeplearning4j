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

#include <helper_cuda.h>
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>
#include "legacy_ops.h"


#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/sharedmem.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif



namespace functions {
    namespace pairwise_transforms {

/**
 * Transforms involving 2 arrays
 */
        template<typename X, typename Y, typename Z>
        class PairWiseTransform {
        public:

#ifdef __CUDACC__


            template<typename OpType>
            static __device__ void transformCuda(Nd4jLong len, void *x, void *y, Nd4jLong xEws, Nd4jLong yEws, void *params, void *z, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);



            static __device__ void transformCuda(const int opNum, Nd4jLong len, void *x, void *y, Nd4jLong xEws, Nd4jLong yEws, void *extraParams, void *z, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __host__ void execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *x, Nd4jLong xStride, void *y, Nd4jLong yStride, void *z, Nd4jLong resultStride, void *extraParams, Nd4jLong len);

            static __host__ void execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *resultShapeInfo, void *extraParams);            

            static __device__ void transformCuda(const int opNum, void *x, Nd4jLong *xShapeBuffer, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *resultShapeBuffer, void *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, void *x, Nd4jLong *xShapeBuffer, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *resultShapeBuffer, void *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transformCuda(void *x, Nd4jLong *xShapeBuffer, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *resultShapeBuffer, void *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);            

            template<typename OpType>
	        static __device__ void transform(void *x, Nd4jLong *xShapeBuffer, void *y, Nd4jLong *yShapeBuffer, void *z, Nd4jLong *resultShapeBuffer, void *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes,  int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


#endif
        public:

            static void exec(
				const int opNum,
				void *x,
				Nd4jLong *xShapeBuffer,
				void *y,
				Nd4jLong *yShapeBuffer,
				void *z,
				Nd4jLong *resultShapeBuffer,
				void *extraParams);
			
			static void exec(
				const int opNum,
				void *x,
				Nd4jLong xStride,
				void *y,
				Nd4jLong yStride,
				void *z,
				Nd4jLong resultStride,
				void *extraParams,
				Nd4jLong len);


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
                             const Nd4jLong len);
        };
    }
}

#endif /* PAIRWISE_TRANSFORM_H_ */
