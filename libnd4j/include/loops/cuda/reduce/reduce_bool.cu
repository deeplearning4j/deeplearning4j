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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <exceptions/cuda_exception.h>
#include <execution/LaunchContext.h>
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <loops/reduce_bool.h>
#include <loops/scalar.h>
#include <system/op_boilerplate.h>
#include <system/common.h>
#include <types/types.h>

    using namespace simdOps;

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
SD_KERNEL void simpleReduce(
    const void* x,
    const sd::LongType* outerXTadShapeInfo,
    const sd::LongType* innerXTadShapeInfo,
    void* extraParams,
    void* vreductionBuffer,
    void* z,
    const sd::LongType* zShapeInfo) {

  functions::reduce::ReduceBoolFunction<X, Z>::template transformCuda<OpType>(
      x,
      outerXTadShapeInfo,
      innerXTadShapeInfo,
      vreductionBuffer,
      extraParams,
      z,
      zShapeInfo);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
SD_KERNEL void simpleScalar(
    const void* x,
    const sd::LongType* xShapeInfo,
    void* extraParams,
    void* z,
    const sd::LongType* zShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    void* reductionBuffer,
    const sd::LongType* tadOnlyShapeInfo) {

  functions::reduce::ReduceBoolFunction<X, Z>::template execScalarCuda<OpType>(
      x,
      xShapeInfo,
      extraParams,
      z,
      zShapeInfo,
      reductionBuffer,
      tadOnlyShapeInfo);
}

namespace functions {
namespace reduce {

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void ReduceBoolFunction<X, Z>::aggregatePartials(
    void* vsPartials,
    sd::LongType tid,
    sd::LongType numItems,
    void* vextraParams) {

  auto sPartials   = reinterpret_cast<Z*>(vsPartials);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  sd::LongType floorPow2 = numItems;
  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) {
      floorPow2 &= (floorPow2 - 1);
    }
    if (tid >= floorPow2) {
      sPartials[tid - floorPow2] =
          OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
    }
    __syncthreads();
  }

  for (sd::LongType activeThreads = (floorPow2 >> 1); activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads && (tid + activeThreads) < numItems) {
      sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void ReduceBoolFunction<X, Z>::transformCuda(
    const void* vx,
    const sd::LongType* outerXTadShapeInfo,
    const sd::LongType* innerXTadShapeInfo,
    void* vreductionBuffer,
    void* vextraParams,
    void* vz,
    const sd::LongType* zShapeInfo) {

  auto x           = reinterpret_cast<const X*>(vx);
  auto z           = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  __shared__ Z sPartials[SD_CUDA_BLOCK_SIZE];
  __shared__ int tadLen;
  __shared__ int numTads;
  __shared__ bool sameOffsets;

  // Cache shape info
  __shared__ sd::LongType outerRank;
  __shared__ sd::LongType innerRank;
  __shared__ sd::LongType zRank;

  __shared__ const sd::LongType* outerShapePtr;
  __shared__ const sd::LongType* outerStridePtr;

  __shared__ const sd::LongType* innerShapePtr;
  __shared__ const sd::LongType* innerStridePtr;

  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    outerRank      = shape::rank(outerXTadShapeInfo);
    outerShapePtr  = shape::shapeOf(outerXTadShapeInfo);
    outerStridePtr = shape::stride(outerXTadShapeInfo);

    innerRank      = shape::rank(innerXTadShapeInfo);
    innerShapePtr  = shape::shapeOf(innerXTadShapeInfo);
    innerStridePtr = shape::stride(innerXTadShapeInfo);

    zRank          = shape::rank(zShapeInfo);
    zShapePtr      = shape::shapeOf(zShapeInfo);
    zStridePtr     = shape::stride(zShapeInfo);

    sameOffsets    = shape::haveSameShapeAndStrides(zShapeInfo, outerXTadShapeInfo);
    tadLen         = shape::length(innerXTadShapeInfo);
    numTads        = shape::length(outerXTadShapeInfo);
  }
  __syncthreads();

  sd::LongType coords[SD_MAX_RANK];

  for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    INDEX2COORDS(r, outerRank, outerShapePtr, coords);

    sd::LongType outerOffset;
    COORDS2INDEX(outerRank, outerStridePtr, coords, outerOffset);

    sd::LongType zOffset;
    if (sameOffsets) {
      zOffset = outerOffset;
    } else {
      INDEX2COORDS(r, zRank, zShapePtr, coords);
      COORDS2INDEX(zRank, zStridePtr, coords, zOffset);
    }

    const X* xTad = x + outerOffset;
    sPartials[threadIdx.x] = OpType::startingValue(xTad);

    for (sd::LongType i = threadIdx.x; i < tadLen; i += blockDim.x) {
      sd::LongType iCoords[SD_MAX_RANK];
      sd::LongType innerOffset;

      INDEX2COORDS(i, innerRank, innerShapePtr, iCoords);
      COORDS2INDEX(innerRank, innerStridePtr, iCoords, innerOffset);

      sPartials[threadIdx.x] = OpType::update(
          sPartials[threadIdx.x],
          OpType::op(xTad[innerOffset], extraParams),
          extraParams);
    }
    __syncthreads();

    aggregatePartials<OpType>(
        sPartials,
        threadIdx.x,
        sd::math::sd_min<int>(blockDim.x, tadLen),
        extraParams);
    __syncthreads();

    if (threadIdx.x == 0) {
      z[zOffset] = OpType::postProcess(sPartials[0], tadLen, extraParams);
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void ReduceBoolFunction<X, Z>::execScalarCuda(
    const void* vx,
    const sd::LongType* xShapeInfo,
    void* vextraParams,
    void* vz,
    const sd::LongType* zShapeInfo,
    void* vreductionBuffer,
    const sd::LongType* tadOnlyShapeInfo) {

  auto x           = reinterpret_cast<const X*>(vx);
  auto z           = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<X*>(vextraParams);

  __shared__ Z sPartials[SD_CUDA_BLOCK_SIZE];
  __shared__ sd::LongType length;

  // Cache shape info
  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;
  auto reductionBuffer = reinterpret_cast<Z *>(vreductionBuffer);

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadIdx.x == 0) {
    length     = shape::length(xShapeInfo);
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);
  }
  __syncthreads();

  sPartials[threadIdx.x] = OpType::startingValue(x);

  sd::LongType gridSize = gridDim.x * blockDim.x;
  for (sd::LongType i = tid; i < length; i += gridSize) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType xOffset;

    INDEX2COORDS(i, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);

    sPartials[threadIdx.x] = OpType::update(
        sPartials[threadIdx.x],
        OpType::op(x[xOffset], extraParams),
        extraParams);
  }
  __syncthreads();

  aggregatePartials<OpType>(
      sPartials,
      threadIdx.x,
      sd::math::sd_min<int>(blockDim.x, length),
      extraParams);
  __syncthreads();

  if (gridDim.x > 1) {
    auto tc = reinterpret_cast<unsigned int*>(vreductionBuffer);
    __shared__ bool amLast;

    if (threadIdx.x == 0) {
      reductionBuffer[blockIdx.x] = sPartials[0];
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
      unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
      amLast              = (ticket == (gridDim.x - 1));
    }
    __syncthreads();

    if (amLast) {
      tc[16384] = 0;
      sPartials[threadIdx.x] = OpType::startingValue(x);

      for (sd::LongType i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
        sPartials[threadIdx.x] = OpType::update(
            sPartials[threadIdx.x],
            reinterpret_cast<Z*>(vreductionBuffer)[i],
            extraParams);
      }
      __syncthreads();

      aggregatePartials<OpType>(
          sPartials,
          threadIdx.x,
          sd::math::sd_min<int>(gridDim.x, blockDim.x),
          extraParams);
      __syncthreads();

      if (threadIdx.x == 0) {
        z[0] = OpType::postProcess(sPartials[0], length, extraParams);
      }
    }
  } else {
    if (threadIdx.x == 0) {
      auto tc   = reinterpret_cast<unsigned int*>(vreductionBuffer);
      tc[16384] = 0;
      z[0]      = OpType::postProcess(sPartials[0], length, extraParams);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_HOST void ReduceBoolFunction<X, Z>::intermediate(
    dim3 launchDims,
    cudaStream_t* stream,
    const void* x,
    sd::LongType* dXShapeInfo,
    sd::LongType* hXShapeInfo,
    void* extraParams,
    void* vreductionBuffer,
    void* z,
    sd::LongType* dZShapeInfo,
    sd::LongType* hZShapeInfo,
    sd::LongType* dims) {

  if (shape::isEmptyConst(hXShapeInfo)) {
    if (shape::isEmptyConst(hZShapeInfo)) return;

    const auto startingVal =
        static_cast<Z>(OpType::startingValue(reinterpret_cast<const X*>(x)));

    auto res = cudaMemcpyAsync(
        sd::LaunchContext::defaultContext()->getScalarPointer(),
        &startingVal,
        sizeof(Z),
        cudaMemcpyHostToDevice,
        *stream);
    if (res != 0) {
      throw sd::cuda_exception::build(
          "ReduceBoolFunction<X,Z>::intermediate: failed to copy temporary scalar", res);
    }

    auto ptr = sd::LaunchContext::defaultContext()->getScalarPointer();
    // scalar assign
    scalar::ScalarTransform<Z,Z,Z>::executeCudaShaped(
        launchDims,
        stream,
        14,
        z,
        dZShapeInfo,
        hZShapeInfo,
        z,
        dZShapeInfo,
        hZShapeInfo,
        ptr,
        nullptr);
    sd::DebugHelper::checkErrorCode(stream, "reduceBoolDim empty(...) failed");
  } else {
    const sd::LongType zRank   = shape::rank(hZShapeInfo);
    const sd::LongType tadRank = shape::rank(hXShapeInfo) - zRank;

    auto outerPack = sd::ConstantShapeHelper::getInstance()
                         .createSubArrShapeInfo(hXShapeInfo, dims, zRank);
    auto innerPack = sd::ConstantShapeHelper::getInstance()
                         .createSubArrShapeInfo(hXShapeInfo, dims + zRank, tadRank);

    simpleReduce<X, Z, OpType>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            x,
            outerPack->special(),
            innerPack->special(),
            extraParams,
            vreductionBuffer,
            z,
            dZShapeInfo);
    sd::DebugHelper::checkErrorCode(stream, "reduceBoolDim(...) failed");
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_HOST void ReduceBoolFunction<X, Z>::intermediateScalar(
    dim3 launchDims,
    cudaStream_t* stream,
    const void* x,
    const sd::LongType* xShapeInfo,
    const sd::LongType* hXShapeInfo,
    void* extraParams,
    void* z,
    const sd::LongType* zShapeInfo,
    const sd::LongType* hZShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    void* reductionBuffer,
    const sd::LongType* tadOnlyShapeInfo) {

  if (shape::isEmptyConst(hXShapeInfo)) {
    if (shape::isEmptyConst(hZShapeInfo)) return;

    const auto startingVal =
        static_cast<Z>(OpType::startingValue(reinterpret_cast<const X*>(x)));
    auto res = cudaMemcpyAsync(
        z,
        &startingVal,
        sizeof(Z),
        cudaMemcpyHostToDevice,
        *stream);
    if (res != 0) {
      throw sd::cuda_exception::build(
          "ReduceBoolFunction<X,Z>::intermediateScalar: failed to copy resulting scalar", res);
    }
    sd::DebugHelper::checkErrorCode(stream, "reduceBoolScalar empty(...) failed");
  } else {
    simpleScalar<X, Z, OpType>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            x,
            xShapeInfo,
            extraParams,
            z,
            zShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer,
            tadOnlyShapeInfo);
    sd::DebugHelper::checkErrorCode(stream, "reduceBoolScalar(...) failed");
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void ReduceBoolFunction<X, Y>::execReduceScalar(
    dim3 launchDims,
    cudaStream_t* stream,
    const int opNum,
    const void* x,
    const sd::LongType* xShapeInfo,
    const sd::LongType* hXShapeInfo,
    void* extraParams,
    void* z,
    const sd::LongType* zShapeInfo,
    const sd::LongType* hZShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    void* reductionBuffer,
    const sd::LongType* tadOnlyShapeInfo) {

  DISPATCH_BY_OPNUM_TT(
      intermediateScalar,
      PARAMS(
          launchDims, stream, x, xShapeInfo, hXShapeInfo, extraParams, z,
          zShapeInfo, hZShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo),
      OPS_A(REDUCE_BOOL_OPS));
  sd::DebugHelper::checkErrorCode(stream, "execReduceScalarFloat(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void ReduceBoolFunction<X, Y>::execReduce(
    dim3 launchDims,
    cudaStream_t* stream,
    const int opNum,
    const void* x,
    sd::LongType* dXShapeInfo,
    sd::LongType* hXShapeInfo,
    void* extraParams,
    void* vreductionBuffer,
    void* z,
    sd::LongType* dZShapeInfo,
    sd::LongType* hZShapeInfo,
    sd::LongType* dims) {

  if (shape::length(hZShapeInfo) == 1) {
    execReduceScalar(
        launchDims, stream, opNum,
        x, dXShapeInfo, hXShapeInfo,
        extraParams, z,
        dZShapeInfo, hZShapeInfo,
        nullptr, 0,
        vreductionBuffer,
        nullptr);
  } else {
    DISPATCH_BY_OPNUM_TT(
        intermediate,
        PARAMS(
            launchDims, stream, x, dXShapeInfo, hXShapeInfo, extraParams,
            vreductionBuffer, z, dZShapeInfo, hZShapeInfo, dims),
        OPS_A(REDUCE_BOOL_OPS));
  }
  DEBUG_KERNEL(stream, opNum);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
SD_DEVICE void initializeShared(X* extraParams, X** sPartials, int sMemSize) {
  int sPartialsLength = sMemSize / sizeof(X);
  X* sPartialsDeref   = reinterpret_cast<X*>(*sPartials);
  for (int i = 0; i < sPartialsLength; i++) {
    sPartialsDeref[i] = extraParams[0];
  }
}

ITERATE_COMBINATIONS(
    (SD_COMMON_TYPES),
    (SD_BOOL_TYPES),
    INSTANT_PROCESS_CLASSCOMBINATION,
     ReduceBoolFunction,
    ;)

}  // namespace reduce
}  // namespace functions
