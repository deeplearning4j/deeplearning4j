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

//
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
#define DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H

#include <ops.h>
#include <types/types.h>
#include <op_boilerplate.h>
#include <shape.h>

using namespace simdOps;

#define LOCAL_TRANSFORM_STRICT_OPS \
        (23, Exp), \
        (24, Log)

namespace functions {
    namespace transform {
        template <typename X>
        class TransformStrictInplace {
        public:
            static FORCEINLINE _CUDA_D void transformCudaLegacy(int opNum, void *dy, Nd4jLong *shapeInfo, void *params, void *result, Nd4jLong *zShapeInfo, int *allocationPointer, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);

            template <typename OpClass>
            static FORCEINLINE _CUDA_D void transformCuda(void *vdy, Nd4jLong *shapeInfo, void *vparams, void *vresult, Nd4jLong *zShapeInfo, int *allocationPointer, void *vreductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets);
        };

        template<typename X>
        template <typename OpType>
        FORCEINLINE _CUDA_D void TransformStrictInplace<X>::transformCuda(
                void *vdy,
                Nd4jLong *shapeInfo,
                void *vparams,
                void *vresult,
                Nd4jLong *zShapeInfo,
                int *allocationPointer, void *vreductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

            auto dy = static_cast<X*>(vdy);
            auto result = static_cast<X*>(vresult);
            auto params = static_cast<X*>(vparams);
            auto reductionPointer = static_cast<X*>(vreductionPointer);

            auto xOrder = shape::order(shapeInfo);
            auto zOrder = shape::order(zShapeInfo);

            auto xEws = shape::elementWiseStride(shapeInfo);
            auto zEws = shape::elementWiseStride(zShapeInfo);
            auto tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ Nd4jLong length;
            if(threadIdx.x == 0)
                length = shape::length(shapeInfo);
            __syncthreads();


            for (Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
                auto xOffset2 = shape::getIndexOffset(i, shapeInfo,  length);
                auto zOffset2 = shape::getIndexOffset(i, zShapeInfo, length);
                result[zOffset2] = OpType::op(dy[xOffset2], params);
            }
        }

        template<typename X>
        FORCEINLINE _CUDA_D void TransformStrictInplace<X>::transformCudaLegacy(
                int opNum,
                void *dy,
                Nd4jLong *shapeInfo,
                void *params,
                void *result,
                Nd4jLong *zShapeInfo,
                int *allocationPointer,
                void *reductionPointer,
                Nd4jLong *tadShapeInfo,
                Nd4jLong *tadOffsets) {
            DISPATCH_BY_OPNUM_T(transformCuda, PARAMS(dy, shapeInfo, params, result, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets), LOCAL_TRANSFORM_STRICT_OPS);
        }
    }
}

#undef LOCAL_TRANSFORM_STRICT_OPS
#endif //DEV_TESTS_TRANSFORM_FLOAT_INPLACE_H
