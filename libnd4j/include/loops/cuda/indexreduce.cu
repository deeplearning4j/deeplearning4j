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
// Created by raver on 4/9/2018.
//
#include <helpers/DebugHelper.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../indexreduce.h"
#include "../legacy_ops.h"

using namespace simdOps;

template <typename X, typename Z>
static SD_KERNEL void simpleIndexReduceGeneric(const int op, void const *dx, sd::LongType const *xShapeInfo,
                                               sd::LongType xRank,
                                               void *extraParams, void *result, sd::LongType const *zShapeInfo, sd::LongType zRank,
                                               sd::LongType *dimension, sd::LongType dimensionLength, int postProcessOrNot, sd::LongType *allocationBuffer, void *reductionBuffer,
                                               sd::LongType const *tadOnlyShapeInfo, sd::LongType const *tadOffsets) {
  functions::indexreduce::IndexReduce<X, Z>::transform(op, dx, xShapeInfo, extraParams, result, zShapeInfo, dimension,
                                                       dimensionLength, postProcessOrNot, allocationBuffer,
                                                       reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

namespace functions {
namespace indexreduce {

template <typename X, typename Z>
SD_HOST void IndexReduce<X, Z>::executeIndexReduceScalar(
    dim3 launchDims, cudaStream_t *stream, const int opNum, void const *dx, sd::LongType const *xShapeInfo,
    sd::LongType xRank,
    void *extraParams, void *result, sd::LongType const *zShapeInfo, sd::LongType zRank,
    sd::LongType *dimension, sd::LongType dimensionLength,
    int postProcessOrNot,sd::LongType *allocationBuffer, void *reductionBuffer, sd::LongType const *tadOnlyShapeInfo,
    sd::LongType const *tadOffsets) {
  simpleIndexReduceGeneric<X, Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      opNum, dx, xShapeInfo, xRank, extraParams, result, zShapeInfo, 0, nullptr, 0, 1, allocationBuffer,
      reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

template <typename X, typename Z>
SD_HOST void IndexReduce<X, Z>::executeIndexReduce(dim3 launchDims,
                                                   cudaStream_t *stream,
                                                   const int opNum,
                                                   void const *dx,
                                                   sd::LongType const *xShapeInfo,
                                                   sd::LongType xRank,
                                                   void *extraParams,
                                                   void *result,
                                                   sd::LongType const *zShapeInfo,
                                                   sd::LongType zRank,
                                                   sd::LongType *dimension,
                                                   sd::LongType dimensionLength,
                                                   int postProcessOrNot,
                                                   sd::LongType *allocationBuffer,
                                                   void *reductionBuffer,
                                                   sd::LongType const *tadOnlyShapeInfo,
                                                   sd::LongType const *tadOffsets) {
  simpleIndexReduceGeneric<X, Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      opNum, dx, xShapeInfo, xRank, extraParams, result, zShapeInfo, zRank, dimension, dimensionLength, postProcessOrNot,
      allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
}

// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedIndexValue {
  // Ensure that we won't compile any un-specialized types
  SD_DEVICE T *getPointer() {
    extern SD_DEVICE void error(void);
    error();
    return 0;
  }
};

// Following are the specializations for the following types.
// int, sd::Unsigned, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedIndexValue<float> {
  SD_DEVICE IndexValue<float> *getPointer() {
    extern __shared__ IndexValue<float> s_int2[];
    return s_int2;
  }
};
// Following are the specializations for the following types.
// int, sd::Unsigned, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedIndexValue<double> {
  SD_DEVICE IndexValue<double> *getPointer() {
    extern __shared__ IndexValue<double> s_int6[];
    return s_int6;
  }
};

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void IndexReduce<X, Z>::aggregatePartials(IndexValue<X> *sPartials, sd::LongType tid,
                                                    sd::LongType numElements, void *vextraParams) {
  // start the shared memory loop on the next power of 2 less
  // than the block size.  If block size is not a power of 2,
  // accumulate the intermediate sums in the remainder range.
  auto extraParams = static_cast<X *>(vextraParams);
  sd::LongType floorPow2 = static_cast<sd::LongType>(blockDim.x);

  //ignore this code block
  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) {
      floorPow2 &= floorPow2 - 1;
    }

    if (tid >= floorPow2) {
      IndexValue<X> prev = sPartials[tid - floorPow2];
      IndexValue<X> curr = sPartials[tid];
      sPartials[tid - floorPow2] = OpType::update(prev, curr, extraParams);
    }
    __syncthreads();
  }

  //ignore this code block
  for (sd::LongType activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads && tid + activeThreads < numElements) {
      IndexValue<X> curr = sPartials[tid];
      IndexValue<X> next = sPartials[tid + activeThreads];
      sPartials[tid] = OpType::update(curr, next, extraParams);
    }
    __syncthreads();
  }
}

template <typename X, typename Y>
SD_DEVICE void IndexReduce<X, Y>::transform(int opNum, void const *x, sd::LongType const *xShapeInfo,
                                            void *extraParams, void *result, sd::LongType const *zShapeInfo, sd::LongType *dimension,
                                            sd::LongType dimensionLength, int postProcessOrNot,
                                            sd::LongType *allocationBuffer, void *reductionBuffer,
                                            sd::LongType const *tadShapeInfo, sd::LongType const *tadOffset) {
  DISPATCH_BY_OPNUM_TT(transform,
                       PARAMS(x, xShapeInfo, extraParams, result, zShapeInfo, dimension, dimensionLength,
                              postProcessOrNot, allocationBuffer, reductionBuffer, tadShapeInfo, tadOffset),
                       INDEX_REDUCE_OPS);
}

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void IndexReduce<X, Z>::transform(void const *vdx, sd::LongType const *xShapeInfo, void *vextraParams,
                                            void *vz, sd::LongType const *zShapeInfo, sd::LongType *dimension,
                                            sd::LongType dimensionLength, int postProcessOrNot,
                                            sd::LongType *allocationBuffer,
                                            void *vreductionBuffer, sd::LongType const *tadOnlyShapeInfo,
                                            sd::LongType const *tadOffsets) {
  auto dx = reinterpret_cast<X const *>(vdx);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = static_cast<X *>(vextraParams);
  auto reductionBuffer = static_cast<unsigned int *>(vreductionBuffer);
  auto order = shape::order(xShapeInfo);
  sd::LongType tid = static_cast<sd::LongType>(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ volatile bool resultScalar;

  __shared__ IndexValue<X> sPartials[SD_CUDA_BLOCK_SIZE];

  sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

  __shared__ volatile sd::LongType xLength;

  __shared__ volatile sd::LongType zLen;

  IndexValue<X> reduction = OpType::startingIndexValue(dx);
  sd::LongType threadIdxX = static_cast<sd::LongType>(threadIdx.x);
  sd::LongType blockDimX = static_cast<sd::LongType>(blockDim.x);
  sd::LongType blockIdxX = static_cast<sd::LongType>(blockIdx.x);
  sd::LongType gridDimX = static_cast<sd::LongType>(gridDim.x);

  if (threadIdxX == 0) {
    if (zShapeInfo != nullptr)
      zLen = shape::length(zShapeInfo);
    else
      zLen = 1;

    if (zLen == 1)
      resultScalar = true;
    else
      resultScalar = false;

    xLength = shape::length(xShapeInfo);
  }
  __syncthreads();

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;

    for (sd::LongType i = blockIdxX * blockDim.x + threadIdxX; i < zLen; i += gridDimX * blockDimX) {
      z[i] = static_cast<Z>(reduction.index);
    }
    return;
  }

  //ignore this code block
  if (!resultScalar) {
    __shared__ sd::LongType tadLength;
    __shared__ sd::LongType tadEWS;
    __shared__ sd::LongType numTads;

    if (threadIdx.x == 0) {
      tadLength = shape::length(tadOnlyShapeInfo);
      tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
      numTads = shape::length(xShapeInfo) / tadLength;
    }
    __syncthreads();

    if (dimensionLength > 1 || tadEWS < 1) {
      for (sd::LongType r = blockIdxX; r < numTads; r += gridDimX) {
        auto tadOffsetForBlock = tadOffsets[r];
        sPartials[threadIdxX] = OpType::startingIndexValue(dx);

        for (sd::LongType i = threadIdxX; i < tadLength; i += blockDimX) {
          auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo);
          IndexValue<X> comp{dx[xOffset], i};
          sPartials[threadIdxX] = OpType::update(sPartials[threadIdxX], comp, extraParams);
        }

        __syncthreads();
        aggregatePartials<OpType>(sPartials,threadIdxX, sd::math::sd_min<sd::LongType>(blockDimX, tadLength), extraParams);

        __syncthreads();
        if (threadIdxX == 0) {
          z[r] = static_cast<Z>(sPartials[threadIdxX].index);
        }
        __syncthreads();
      }
    } else {
      for (sd::LongType i = blockIdxX; i < numTads; i += gridDimX) {
        sd::LongType tadOffsetForBlock = tadOffsets[i];

        sPartials[threadIdxX] = OpType::startingIndexValue(dx);

        for (sd::LongType x = threadIdxX; x < tadLength; x += blockDimX) {
          IndexValue<X> comp{dx[tadOffsetForBlock + x * tadEWS], x};
          sPartials[threadIdxX] = OpType::update(sPartials[threadIdxX], comp, extraParams);
        }

        __syncthreads();
        aggregatePartials<OpType>(sPartials, threadIdxX, sd::math::sd_min<sd::LongType>(blockDim.x, tadLength), extraParams);

        __syncthreads();
        if (threadIdxX == 0) {
          z[i] = static_cast<Z>(sPartials[threadIdxX].index);
        }
        __syncthreads();
      }
    }
  } else {
    auto n = shape::length(xShapeInfo);
    auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);
    if (xElementWiseStride >= 1 && order == 'c') {
    //  printf("xEleStride > 1 && order == c\n");
      for (sd::LongType i = tid; i < n; i += (gridDimX * blockDimX)) {
        IndexValue<X> comp{dx[i * xElementWiseStride], i};
        reduction = OpType::update(reduction, comp, extraParams);
      }


    } else {
      for (sd::LongType i = tid; i < n; i += (gridDimX * blockDimX)) {
        auto xOffset = shape::getIndexOffset(i, xShapeInfo);
        IndexValue<X> comp{dx[xOffset], i};
        reduction = OpType::update(reduction, comp, extraParams);
      }


    }
    sPartials[threadIdxX] = reduction;
    __syncthreads();
    aggregatePartials<OpType>(sPartials, threadIdxX, sd::math::sd_min<sd::LongType>(blockDim.x, n), extraParams);
    //printf("After aggregate partials\n");
    if (gridDimX > 1) {
     // printf("grimdDimX > 1\n");
      __shared__ bool amLast;
      unsigned int *unsignedSharedMemory = (unsigned int *)reductionBuffer;
      tid = threadIdx.x;
      if (threadIdx.x == 0)
        reductionBuffer[blockIdx.x] = sPartials[threadIdx.x].index;

      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0) {
        unsigned int ticket = atomicInc(&unsignedSharedMemory[16384], gridDim.x);
        amLast = (ticket == gridDim.x - 1);
      }

      __syncthreads();

      if (amLast) {
        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);
        for (sd::LongType i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
          IndexValue<X> comp{static_cast<X>(0), reductionBuffer[i]};
          sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], comp, extraParams);
        }
        __syncthreads();
        aggregatePartials<OpType>(sPartials, threadIdxX, gridDim.x, extraParams);

        if (threadIdx.x == 0) {
          z[0] = static_cast<Z>(sPartials[threadIdx.x].index);
          unsignedSharedMemory[16384] = 0;
        }
      }
    } else {
     // printf("grimdDimX < 1\n" );
      if (threadIdx.x == 0) {
        z[0] = static_cast<Z>(sPartials[threadIdx.x].index);
       // printf("z[0] %f\n", z[0]);
      }

      //printf("After imdDimX < 1\n" );
    }
  }
}

BUILD_DOUBLE_TEMPLATE(template class IndexReduce, , SD_COMMON_TYPES, SD_INDEXING_TYPES);
}  // namespace indexreduce
}  // namespace functions
