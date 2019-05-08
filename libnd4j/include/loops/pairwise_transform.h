/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include <dll.h>
#include <stdio.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include <types/types.h>
#include "legacy_ops.h"
#include <helpers/DebugHelper.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <types/float16.h>
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

            template <typename OpType>            
            static __host__ void intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vy, Nd4jLong *yShapeInfo, Nd4jLong *hyShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void *vextraParams, int *allocationPointer);

            static __host__ void executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *y, Nd4jLong *yShapeInfo, Nd4jLong *hyShapeInfo, void *z, Nd4jLong *resultShapeInfo, Nd4jLong *hzShapeInfo, void *extraParams);

#endif
        public:

            static void exec(
				const int opNum,
				void *x,
				Nd4jLong *xShapeInfo,
				void *y,
				Nd4jLong *yShapeInfo,
				void *z,
				Nd4jLong *zShapeInfo,
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
                    Nd4jLong* xShapeInfo,
                    void *vy,
                    Nd4jLong* yShapeInfo,
                    void *vresult,
                    Nd4jLong* zShapeInfo,
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
