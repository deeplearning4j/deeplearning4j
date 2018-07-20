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
// @author raver119@gmail.com
//
//
#ifndef LIBND4J_GRID_H
#define LIBND4J_GRID_H

#include <ops/ops.h>
#include <types/float16.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace functions {
    namespace grid {
        template <typename T>
        class GRIDShaped {
        public:
            static void execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *dz, Nd4jLong *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB);

            template<typename OpType>
            static __device__ void transformCuda(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(int opTypeA, int opNumA, int opTypeB, int opNumB,  T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);
        };
    }
}

#endif //LIBND4J_GRID_H
