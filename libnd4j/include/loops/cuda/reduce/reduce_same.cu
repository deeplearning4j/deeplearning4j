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
#include <loops/reduce_same.h>
#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>
#include <types/types.h>
#include <execution/LaunchContext.h>
#include <exceptions/cuda_exception.h>
#include <loops/scalar.h>


using namespace simdOps;


////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ void simpleReduce(void *x, Nd4jLong *xShapeInfo,
                            void *extraParams,
                            void *z, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            void *reductionBuffer,
                            Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    functions::reduce::ReduceSameFunction<X>::template transformCudaXD<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ void simpleScalar(void *x, Nd4jLong *xShapeInfo,
                            void *extraParams,
                            void *z, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

    functions::reduce::ReduceSameFunction<X>::template execScalarCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, reductionBuffer, tadOnlyShapeInfo);
}


namespace functions {
namespace reduce    {

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ReduceSameFunction<X>::aggregatePartials(void *vsPartials, Nd4jLong tid, Nd4jLong numItems, void *vextraParams) {

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

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ReduceSameFunction<X>::transformCudaXD( void *vx, Nd4jLong *xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong *zShapeInfo,
                                                        int *dimension,  int dimensionLength,
                                                        void *vreductionBuffer,
                                                        Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<X*>(vx);
    auto z = reinterpret_cast<X*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);
    auto reductionBuffer = reinterpret_cast<X*>(vreductionBuffer);

    if (OpType::requiresSpecialAccumulation) {
        OpType::execSpecialCuda(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
        return;
    }

    //shared memory space for storing intermediate results
    __shared__ X* sPartials;

    __shared__ int tadLength, tadRank, numTads;
    __shared__ Nd4jLong *tadShape, *tadStride;
    __shared__ bool isPlainOutput;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<X*>(shmem);

        isPlainOutput = shape::order(zShapeInfo) == 'c' && shape::elementWiseStride(zShapeInfo) == 1;

        tadLength = shape::length(tadOnlyShapeInfo);
        tadRank = shape::rank(tadOnlyShapeInfo);
        numTads = shape::length(xShapeInfo) / tadLength;
        tadShape = shape::shapeOf(tadOnlyShapeInfo);
        tadStride = shape::stride(tadOnlyShapeInfo);
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
template <typename X>
__device__ void ReduceSameFunction<X>::execScalarCudaLegacy(int opNum, void *vx, Nd4jLong *xShapeInfo,
                                                          void *vextraParams,
                                                          void *vz, Nd4jLong *zShapeInfo,
                                                          void *vreductionBuffer,
                                                          Nd4jLong *tadOnlyShapeInfo) {
    DISPATCH_BY_OPNUM_T(execScalarCuda, PARAMS(vx, xShapeInfo, vextraParams, vz, zShapeInfo, vreductionBuffer, tadOnlyShapeInfo), REDUCE_SAME_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ReduceSameFunction<X>::execScalarCuda(void *vx, Nd4jLong *xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong *zShapeInfo,
                                                        void *vreductionBuffer,
                                                        Nd4jLong *tadOnlyShapeInfo) {
    auto x = reinterpret_cast<X*>(vx);
    auto z = reinterpret_cast<X*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);
    auto reductionBuffer = reinterpret_cast<X*>(vreductionBuffer);

    auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    //shared memory space for storing intermediate results
    __shared__ X* sPartials;
    __shared__ Nd4jLong xEws;
    __shared__ Nd4jLong len;

    if(threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sPartials = reinterpret_cast<X*>(shmem);
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
            auto tc = reinterpret_cast<unsigned int *>(reductionBuffer);
            tc[16384] = 0;
            z[0] = OpType::postProcess(sPartials[0], len, extraParams);
        }
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpType>
__host__ void ReduceSameFunction<X>::intermediateXD(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong *hZShapeInfo, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    if(shape::isEmpty(hXShapeInfo)) {

        if(shape::isEmpty(hZShapeInfo))
            return;

        const auto startingVal = static_cast<X>(OpType::startingValue(reinterpret_cast<X*>(x)));

        auto res = cudaMemcpyAsync(nd4j::LaunchContext::defaultContext()->getScalarPointer(), &startingVal, sizeof(X), cudaMemcpyHostToDevice, *stream);
        if (res != 0)
            throw nd4j::cuda_exception::build("ReduceSameFunction<X,Z>::intermediateXD: failed to copy temporary scalar", res);

        auto ptr = nd4j::LaunchContext::defaultContext()->getScalarPointer();

        // scalar assign
        functions::scalar::ScalarTransform<X, X, X>::executeCudaShaped(launchDims, stream, 14, z, zShapeInfo, hXShapeInfo, z, zShapeInfo, hZShapeInfo, ptr, nullptr);
    }
    else {
        simpleReduce<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets);
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpType>
__host__ void ReduceSameFunction<X>::intermediateScalar(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, Nd4jLong *hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong *hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

    if (shape::isEmpty(hXShapeInfo)) {

        if (shape::isEmpty(hZShapeInfo))
            return;

        const auto startingVal = static_cast<X>(OpType::startingValue(reinterpret_cast<X*>(x)));

        auto res = cudaMemcpyAsync(z, &startingVal, sizeof(X), cudaMemcpyHostToDevice, *stream);
        if (res != 0)
            throw nd4j::cuda_exception::build("ReduceSameFunction<X>::intermediateScalar: failed to copy resulting scalar", res);
    }
    else {
        simpleScalar<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo);
    }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
_CUDA_H void ReduceSameFunction<X>::execReduceScalar(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, Nd4jLong* hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong* hZShapeInfo, int *dimension, int dimensionLength, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo) {

        DISPATCH_BY_OPNUM_T(intermediateScalar, PARAMS(launchDims, stream, x, xShapeInfo, hXShapeInfo, extraParams, z, zShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), REDUCE_SAME_OPS);
        nd4j::DebugHelper::checkErrorCode(stream, "execReduceScalarSame(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X>
_CUDA_H void ReduceSameFunction<X>::execReduceXD(dim3 launchDims, cudaStream_t *stream, int opNum, int rank, void *x, Nd4jLong *xShapeInfo, Nd4jLong* hXShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, Nd4jLong* hZShapeInfo, int *dimension, int dimensionLength, void *reductionPointer, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    DISPATCH_BY_OPNUM_T(intermediateXD, PARAMS(launchDims, stream, x, xShapeInfo, hXShapeInfo, extraParams, z, zShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), REDUCE_SAME_OPS);
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


BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ReduceSameFunction, , LIBND4J_TYPES);

}
}