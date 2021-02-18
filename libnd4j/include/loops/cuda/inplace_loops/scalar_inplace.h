/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//


#ifndef DEV_TESTS_SCALAR_INPLACE_H
#define DEV_TESTS_SCALAR_INPLACE_H

#include <ops.h>
#include <types/types.h>
#include <system/op_boilerplate.h>
#include <helpers/shape.h>

using namespace simdOps;

namespace functions {
    namespace scalar {
        template <typename X, typename Y, typename Z>
        class ScalarInplace {
        public:
            static FORCEINLINE _CUDA_D void transformCudaLegacy(int opNum, void* vscalar, void *vy, Nd4jLong *yShapeInfo, void *vparams, void *vz, Nd4jLong *zShapeInfo, int *allocationBuffer);

            template <typename OpClass>
            static FORCEINLINE _CUDA_D void transformCuda(void* vscalar, void *vy, Nd4jLong *yShapeInfo, void *vparams, void *vz, Nd4jLong *zShapeInfo, int *allocationBuffer);
        };

        template<typename X, typename Y, typename Z>
        FORCEINLINE _CUDA_D void ScalarInplace<X,Y,Z>::transformCudaLegacy(int opNum, void* vscalar,
                                                                    void *vy, Nd4jLong *yShapeInfo,
                                                                    void *vparams,
                                                                    void *vz, Nd4jLong *zShapeInfo,
                                                                    int *allocationBuffer) {

            DISPATCH_BY_OPNUM_TTT(transformCuda, PARAMS(vscalar, vy, yShapeInfo, vparams, vz, zShapeInfo, allocationBuffer), SCALAR_OPS);
        }

        template<typename X, typename Y, typename Z>
        template<typename OpType>
        FORCEINLINE _CUDA_D void ScalarInplace<X,Y,Z>::transformCuda(void* vscalar,
                                                              void *vy, Nd4jLong *yShapeInfo,
                                                              void *vparams,
                                                              void *vz, Nd4jLong *zShapeInfo,
                                                              int *allocationBuffer) {

            auto scalar = reinterpret_cast<X*>(vscalar)[0];
            auto y      = reinterpret_cast<Y*>(vy);
            auto params = reinterpret_cast<Z*>(vparams);
            auto z = reinterpret_cast<Z*>(vz);

            int totalThreads = gridDim.x * blockDim.x;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ Nd4jLong length;
            if(threadIdx.x == 0)
                length = shape::length(yShapeInfo);
            __syncthreads();


            for (Nd4jLong i = tid; i < length; i+= totalThreads) {
                z[shape::getIndexOffset(i, zShapeInfo)] = OpType::op(y[shape::getIndexOffset(i, yShapeInfo)], scalar, params);
            }
        }
    }
}

#endif //DEV_TESTS_SCALAR_INPLACE_H
