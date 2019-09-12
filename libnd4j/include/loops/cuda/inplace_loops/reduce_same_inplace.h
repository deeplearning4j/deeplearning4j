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
//  @author raver119@gmail.com
//


#ifndef DEV_TESTS_REDUCE_SAME_LOOPS_H
#define DEV_TESTS_REDUCE_SAME_LOOPS_H

#include <ops.h>
#include <types/types.h>
#include <op_boilerplate.h>
#include <shape.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X>
        class ReduceSameInplace {
        public:
            static FORCEINLINE void _CUDA_D execScalarCudaLegacy(int opNum, void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, void *vreductionBuffer, Nd4jLong *tadOnlyShapeInfo);

            template <typename OpClass>
            static FORCEINLINE void _CUDA_D execScalarCuda(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, void *vreductionBuffer, Nd4jLong *tadOnlyShapeInfo);

            template <typename OpClass>
            static FORCEINLINE void _CUDA_D aggregatePartials(void *vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams);
        };

        template <typename X>
        template <typename OpType>
        __device__ void ReduceSameInplace<X>::aggregatePartials(void *vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams) {

            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.

            auto sPartials = static_cast<X*>(vsPartials);
            auto extraParams = static_cast<X*>(vextraParams);

            Nd4jLong floorPow2 = numItems;

            if (floorPow2 & (floorPow2 - 1)) {

                while (floorPow2 & (floorPow2 - 1))
                    floorPow2 &= floorPow2 - 1;

                if (tid >= floorPow2)
                    sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);

                __syncthreads();
            }

            for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads && tid + activeThreads < numItems)
                    sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);

                __syncthreads();
            }
        }

        template <typename X>
        FORCEINLINE void _CUDA_D ReduceSameInplace<X>::execScalarCudaLegacy(int opNum, void *vx, Nd4jLong *xShapeInfo,
                                                                    void *vextraParams,
                                                                    void *vz, Nd4jLong *zShapeInfo,
                                                                    void *vreductionBuffer,
                                                                    Nd4jLong *tadOnlyShapeInfo) {
            DISPATCH_BY_OPNUM_T(execScalarCuda, PARAMS(vx, xShapeInfo, vextraParams, vz, zShapeInfo, vreductionBuffer, tadOnlyShapeInfo), REDUCE_SAME_OPS);
        }

        template <typename X>
        template <typename OpType>
        FORCEINLINE void _CUDA_D ReduceSameInplace<X>::execScalarCuda(void *vx, Nd4jLong *xShapeInfo,
                                                              void *vextraParams,
                                                              void *vz, Nd4jLong *zShapeInfo,
                                                              void *vreductionBuffer,
                                                              Nd4jLong *tadOnlyShapeInfo) {

            auto x = reinterpret_cast<X*>(vx);
            auto z = reinterpret_cast<X*>(vz);
            auto extraParams = reinterpret_cast<X*>(vextraParams);
            auto reductionBuffer = reinterpret_cast<X*>(vreductionBuffer);

            int xEws = shape::elementWiseStride(xShapeInfo);
            auto len = shape::length(xShapeInfo);
            auto tid = blockDim.x * blockIdx.x + threadIdx.x;

            //shared memory space for storing intermediate results
            __shared__ X* sPartials;
            if(threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                sPartials = reinterpret_cast<X*>(shmem);
            }
            __syncthreads();
            sPartials[threadIdx.x] = OpType::startingValue(x);

            if (xEws > 0)
                for (int i = tid; i < len; i += (blockDim.x * gridDim.x))
                    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[i * xEws], extraParams), extraParams);
            else
                for (int i = tid; i < len; i += blockDim.x * gridDim.x)
                    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[shape::getIndexOffset(i, xShapeInfo)], extraParams), extraParams);

            __syncthreads();
            aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, len), extraParams);
            __syncthreads();


            if (gridDim.x > 1) {

                unsigned int *tc = (unsigned int *)reductionBuffer;
                __shared__ bool amLast;

                tid = threadIdx.x;
                if (threadIdx.x == 0)
                    reductionBuffer[blockIdx.x] = sPartials[0];//this->postProcess(sPartials[0],len,extraParams);

                __threadfence();
                __syncthreads();

                if (threadIdx.x == 0) {
                    unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                    amLast = (ticket == gridDim.x - 1);
                }

                __syncthreads();

                if (amLast) {

                    tc[16384] = 0;
                    sPartials[threadIdx.x] = OpType::startingValue(x);

                    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
                        sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraParams);

                    __syncthreads();
                    aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraParams);
                    __syncthreads();

                    if (threadIdx.x == 0) {
                        z[0] = OpType::postProcess(sPartials[0], len, extraParams);
                    }
                }
            }
            else {

                if (threadIdx.x == 0) {
                    unsigned int *tc = (unsigned *)reductionBuffer;
                    tc[16384] = 0;
                    z[0] = OpType::postProcess(sPartials[0], len, extraParams);
                }
            }
        }
    }
}

#endif //DEV_TESTS_REDUCE_SAME_LOOPS_H
