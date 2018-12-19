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
// Created by raver119 on 10.10.17.
//

#ifndef PROJECT_GRID_STRIDED_H
#define PROJECT_GRID_STRIDED_H


namespace functions {
    namespace grid {
        template <typename T>
        class GRIDStrided {
        public:
            static void execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, T *dx, Nd4jLong xStride, T *dy, Nd4jLong yStride, T *dz, Nd4jLong zStride, T *extraA, T *extraB, T scalarA, T scalarB);

            template<typename OpType>
            static __device__ void transformCuda(Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz,int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz,int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);
        };
    }
}

#endif //PROJECT_GRID_STRIDED_H
