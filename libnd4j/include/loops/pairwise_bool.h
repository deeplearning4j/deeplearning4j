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

#ifndef PAIRWISE_BOOL_H_
#define PAIRWISE_BOOL_H_
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
#include <helpers/DebugHelper.h>

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
        template<typename X, typename Z>
        class PairWiseBoolTransform {
        public:

#ifdef __CUDACC__

            template <typename OpType>            
            static __host__ void intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, void *vextraParams, int *allocationPointer);

            template<typename OpType>
            static __device__ void transformCuda(Nd4jLong len, void *x, void *y, Nd4jLong xEws, Nd4jLong yEws, void *params, void *z, Nd4jLong zEws, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
            static __device__ void transformCuda(void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, void *extraParams, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, Nd4jLong len, void *x, void *y, Nd4jLong xEws, Nd4jLong yEws, void *extraParams, void *z, Nd4jLong zEws, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, void *extraParams, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo);

            static __host__ void executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *resultShapeInfo, void *extraParams);


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
