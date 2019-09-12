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
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#include <loops/reduce_long.h>
#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>
#include <types/types.h>
#include <execution/LaunchContext.h>
#include <exceptions/cuda_exception.h>
#include <loops/scalar.h>


using namespace simdOps;

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__device__ void reduceSimpleGeneric(void *x,  Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    void *z, Nd4jLong *zShapeInfo,
                                    int *dimension, int dimensionLength,
                                    void *reductionBuffer,
                                    Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    functions::reduce::ReduceLongFunction<X,Z>::template transformCudaXD<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__device__ void reduceScalarGeneric(void *x, Nd4jLong *xShapeInfo,
                                    void *extraParams,
                                    void *z, Nd4jLong *zShapeInfo,
                                    int *dimension, int dimensionLength,
                                    void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

    functions::reduce::ReduceLongFunction<X, Z>::template execScalarCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, reductionBuffer, tadOnlyShapeInfo);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__global__ void simpleReduce(void *x, Nd4jLong *xShapeInfo,
                            void *extraParams,
                            void *z, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            void *reductionBuffer,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    reduceSimpleGeneric<X, Z, OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
__global__ void simpleScalar(void *x, Nd4jLong *xShapeInfo,
                            void *extraParams,
                            void *z, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

    reduceScalarGeneric<X, Z, OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo);
}

namespace functions {
namespace reduce    {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__device__ void ReduceLongFunction<X,Z>::aggregatePartials(void *vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams) {

    // start the shared memory loop on the next power of 2 less
    // than the block size.  If block size is not a power of 2,
    // accumulate the intermediate sums in the remainder range.

    auto sPartials = reinterpret_cast<Z*>(vsPartials);
    auto extraParams = reinterpret_cast<X*>(vextraParams);

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

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__device__ void ReduceLongFunction<X,Z>::transformCudaXD( void *vx, Nd4jLong *xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong *zShapeInfo,
                                                        int *dimension,  int dimensionLength,
                                                        void *vreductionBuffer,
                                                        Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<X*>(vx);
    auto z = reinterpret_cast<Z*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);
    auto reductionBuffer = reinterpret_cast<Z*>(vreductionBuffer);

    //shared memory space for storing intermediate results
    __shared__ Z* sPartials;
    __shared__ int tadLength, numTads;
    __shared__ bool isPlainOutput;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<Z*>(shmem);

        isPlainOutput = shape::order(zShapeInfo) == 'c' && shape::elementWiseStride(zShapeInfo) == 1;

        tadLength = shape::length(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {

        Nd4jLong tadOffsetForBlock = tadOffsets[r];
        sPartials[threadIdx.x] = OpType::startingValue(x + tadOffsetForBlock);

        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
            auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo);
            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(x[xOffset], extraParams), extraParams);
        }
        __syncthreads();

        // aggregate. do NOT reduce for elements > tadLength
        aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);
        __syncthreads();

        if (threadIdx.x == 0)
            z[isPlainOutput ? r : shape::getIndexOffset(r, zShapeInfo)] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__device__ void ReduceLongFunction<X,Z>::execScalarCuda(void *vx, Nd4jLong *xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong *zShapeInfo,
                                                        void *vreductionBuffer,
                                                        Nd4jLong *tadOnlyShapeInfo) {

    auto x = reinterpret_cast<X*>(vx);
    auto z = reinterpret_cast<Z*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);
    auto reductionBuffer = reinterpret_cast<Z*>(vreductionBuffer);

    auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    //shared memory space for storing intermediate results
    __shared__ Z* sPartials;
    __shared__ Nd4jLong xEws;
    __shared__ Nd4jLong len;

    if(threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<Z*>(shmem);
        xEws = shape::elementWiseStride(xShapeInfo);
        len = shape::length(xShapeInfo);
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

        auto tc = reinterpret_cast<unsigned int *>(reductionBuffer);
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
            auto tc = reinterpret_cast<unsigned int*>(reductionBuffer);
            tc[16384] = 0;
            z[0] = OpType::postProcess(sPartials[0], len, extraParams);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
__host__ void ReduceLongFunction<X,Z>::intermediateXD(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong *hZShapeInfo, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    if(shape::isEmpty(hXShapeInfo)) {

        if(shape::isEmpty(hZShapeInfo))
            return;

        const auto startingVal = static_cast<Z>(OpType::startingValue(reinterpret_cast<X*>(x)));

        auto res = cudaMemcpyAsync(nd4j::LaunchContext::defaultContext()->getScalarPointer(), &startingVal, sizeof(Z), cudaMemcpyHostToDevice, *stream);
        if (res != 0)
            throw nd4j::cuda_exception::build("ReduceLongFunction<X,Z>::intermediateXD: failed to copy temporary scalar", res);

        auto ptr = nd4j::LaunchContext::defaultContext()->getScalarPointer();

        // scalar assign
        functions::scalar::ScalarTransform<Z, Z, Z>::executeCudaShaped(launchDims, stream, 14, z, zShapeInfo, hXShapeInfo, z, zShapeInfo, hZShapeInfo, ptr, nullptr);
    }
    else {
        simpleReduce<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets);
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
__host__ void ReduceLongFunction<X,Z>::intermediateScalar(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong *hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

    if (shape::isEmpty(hXShapeInfo)) {

        if (shape::isEmpty(hZShapeInfo))
            return;

        const auto startingVal = static_cast<Z>(OpType::startingValue(reinterpret_cast<X*>(x)));

        auto res = cudaMemcpyAsync(z, &startingVal, sizeof(Z), cudaMemcpyHostToDevice, *stream);
        if (res != 0)
            throw nd4j::cuda_exception::build("ReduceLongFunction<X,Z>::intermediateScalar: failed to copy resulting scalar", res);
    }
    else {
        simpleScalar<X, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo);
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
_CUDA_H void ReduceLongFunction<X,Y>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, Nd4jLong* hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong* hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

        DISPATCH_BY_OPNUM_TT(intermediateScalar, PARAMS(launchDims, stream, x, xShapeInfo, hXShapeInfo, extraParams, z, zShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_LONG_OPS));
        nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarFloat(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
_CUDA_H void ReduceLongFunction<X,Y>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, void *x, Nd4jLong *xShapeInfo, Nd4jLong* hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong* hZShapeInfo, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    DISPATCH_BY_OPNUM_TT(intermediateXD, PARAMS(launchDims, stream, x, xShapeInfo, hXShapeInfo, extraParams, z, zShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_LONG_OPS));
    DEBUG_KERNEL(stream, opNum);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
__device__ void initializeShared(X *extraParams, X **sPartials, int sMemSize) {
    int sPartialsLength = sMemSize / sizeof(X);
    X *sPartialsDeref = (X *) *sPartials;
    for (int i = 0; i < sPartialsLength; i++)
        sPartialsDeref[i] = extraParams[0];

}


BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReduceLongFunction, , LIBND4J_TYPES, LONG_TYPES);

}
}

