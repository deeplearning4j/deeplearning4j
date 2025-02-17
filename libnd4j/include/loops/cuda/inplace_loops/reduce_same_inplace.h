/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_REDUCE_SAME_LOOPS_H
#define DEV_TESTS_REDUCE_SAME_LOOPS_H

#include <cuda_runtime.h>
#include <helpers/shape.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include <execution/LaunchContext.h>
#include "execution/cuda/LaunchDims.h"
using namespace simdOps;

namespace functions {
namespace reduce {

template <typename X>
class ReduceSameInplace {
 public:
  // Dispatch method for old-style calls
  static SD_INLINE void SD_DEVICE execScalarCudaLegacy(
      int opNum,
      void* vx,
      sd::LongType* xShapeInfo,
      void* vextraParams,
      void* vz,
      sd::LongType* zShapeInfo,
      void* vreductionBuffer,
      sd::LongType* tadOnlyShapeInfo);

  // Template for real call
  template <typename OpClass>
  static SD_INLINE void SD_DEVICE execScalarCuda(
      void* vx,
      sd::LongType* xShapeInfo,
      void* vextraParams,
      void* vz,
      sd::LongType* zShapeInfo,
      void* vreductionBuffer,
      sd::LongType* tadOnlyShapeInfo);

  template <typename OpClass>
  static SD_INLINE void SD_DEVICE aggregatePartials(
      void* vsPartials,
      sd::LongType tid,
      sd::LongType numItems,
      void* vextraParams);
};

template <typename X>
template <typename OpClass>
SD_INLINE void SD_DEVICE ReduceSameInplace<X>::aggregatePartials(
    void* vsPartials,
    sd::LongType tid,
    sd::LongType numItems,
    void* vextraParams) {

  auto sPartials   = static_cast<X*>(vsPartials);
  auto extraParams = static_cast<X*>(vextraParams);

  sd::LongType floorPow2 = numItems;
  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) {
      floorPow2 &= (floorPow2 - 1);
    }
    if (tid >= floorPow2) {
      sPartials[tid - floorPow2] =
          OpClass::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
    }
    __syncthreads();
  }

  for (sd::LongType activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads && (tid + activeThreads) < numItems) {
      sPartials[tid] =
          OpClass::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
    }
    __syncthreads();
  }
}

template <typename X>
SD_INLINE void SD_DEVICE ReduceSameInplace<X>::execScalarCudaLegacy(
    int opNum,
    void* vx,
    sd::LongType* xShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType* zShapeInfo,
    void* vreductionBuffer,
    sd::LongType* tadOnlyShapeInfo) {

  DISPATCH_BY_OPNUM_T(
      execScalarCuda,
      PARAMS(vx, xShapeInfo, vextraParams, vz, zShapeInfo, vreductionBuffer, tadOnlyShapeInfo),
      REDUCE_SAME_OPS);
}

template <typename X>
template <typename OpClass>
SD_INLINE void SD_DEVICE ReduceSameInplace<X>::execScalarCuda(
    void* vx,
    sd::LongType* xShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType* zShapeInfo,
    void* vreductionBuffer,
    sd::LongType* tadOnlyShapeInfo) {

  auto x             = reinterpret_cast<X*>(vx);
  auto z             = reinterpret_cast<X*>(vz);
  auto extraParams   = reinterpret_cast<X*>(vextraParams);
  auto reductionBuff = reinterpret_cast<X*>(vreductionBuffer);

  // We'll cache relevant shape info in shared memory so we don't call them repeatedly
  __shared__ sd::LongType length;
  __shared__ int rank;
  __shared__ const sd::LongType* shapePtr;
  __shared__ const sd::LongType* stridePtr;

  if (threadIdx.x == 0) {
    length    = shape::length(xShapeInfo);
    rank      = shape::rank(xShapeInfo);
    shapePtr  = shape::shapeOf(xShapeInfo);
    stridePtr = shape::stride(xShapeInfo);
  }
  __syncthreads();

  const auto tid      = blockIdx.x * blockDim.x + threadIdx.x;
  const auto gridSize = gridDim.x * blockDim.x;

  // We'll use some shared memory for partial sums
  __shared__ X* sPartials;
  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sPartials = reinterpret_cast<X*>(shmem);
  }
  __syncthreads();

  // Each thread gets a starting value
  sPartials[threadIdx.x] = OpClass::startingValue(x);

  // We'll stride over the entire array
  for (sd::LongType i = tid; i < length; i += gridSize) {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType offset;

    INDEX2COORDS(i, rank, shapePtr, coords);
    COORDS2INDEX(rank, stridePtr, coords, offset);

    sPartials[threadIdx.x] =
        OpClass::update(sPartials[threadIdx.x], OpClass::op(x[offset], extraParams), extraParams);
  }
  __syncthreads();

  // Next: reduce partial sums in the block
  aggregatePartials<OpClass>(
      sPartials,
      threadIdx.x,
      sd::math::sd_min<int>(blockDim.x, length),
      extraParams);
  __syncthreads();

  // If gridDim.x > 1, we do a multi-block reduce using the global buffer
  if (gridDim.x > 1) {
    auto tc = reinterpret_cast<unsigned int*>(reductionBuff);
    __shared__ bool amLast;

    // each block's sum is stored in the 'reductionBuff'
    if (threadIdx.x == 0) {
      reductionBuff[blockIdx.x] = sPartials[0];
    }
    __threadfence();
    __syncthreads();

    // The 16384 is a special "counter" location in the reductionBuff
    if (threadIdx.x == 0) {
      unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
      amLast              = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (amLast) {
      // We do a final block-level reduce
      tc[16384] = 0; // reset
      sPartials[threadIdx.x] = OpClass::startingValue(x);

      // accumulate partial sums from each block
      for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += blockDim.x) {
        sPartials[threadIdx.x] =
            OpClass::update(sPartials[threadIdx.x], reductionBuff[i], extraParams);
      }
      __syncthreads();

      aggregatePartials<OpClass>(
          sPartials,
          threadIdx.x,
          sd::math::sd_min<int>(gridDim.x, blockDim.x),
          extraParams);
      __syncthreads();

      if (threadIdx.x == 0) {
        z[0] = OpClass::postProcess(sPartials[0], length, extraParams);
      }
    }
  }
  else {
    // single-block case
    if (threadIdx.x == 0) {
      auto tc   = reinterpret_cast<unsigned int*>(reductionBuff);
      tc[16384] = 0;
      z[0]      = OpClass::postProcess(sPartials[0], length, extraParams);
    }
  }
}

}  // namespace reduce
}  // namespace functions

#endif  // DEV_TESTS_REDUCE_SAME_LOOPS_H
