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
* WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
//  @author raver119@gmail.com
//
#include <helpers/DebugHelper.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <array/ArrayOptions.hXX> // note: keep this. It's required for proper linker work

#include "../indexreduce.h"
#include "../legacy_ops.h"

using namespace simdOps;

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static SD_KERNEL void simpleIndexReduceGeneric(
   const int op,
   void const* dx,
   sd::LongType const* xShapeInfo,
   sd::LongType xRank,
   void* extraParams,
   void* result,
   sd::LongType const* zShapeInfo,
   sd::LongType zRank,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   int postProcessOrNot,
   sd::LongType* allocationBuffer,
   void* reductionBuffer,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets) {

 functions::indexreduce::IndexReduce<X,Z>::transform(
     op, dx, xShapeInfo, extraParams, result, zShapeInfo,
     dimension, dimensionLength, postProcessOrNot,
     allocationBuffer, reductionBuffer,
     tadOnlyShapeInfo, tadOffsets);
}

namespace functions {
namespace indexreduce {

template <typename X, typename Z>
SD_HOST void IndexReduce<X,Z>::executeIndexReduceScalar(
   dim3 launchDims,
   cudaStream_t* stream,
   const int opNum,
   void const* dx,
   sd::LongType const* xShapeInfo,
   sd::LongType xRank,
   void* extraParams,
   void* result,
   sd::LongType const* zShapeInfo,
   sd::LongType zRank,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   int postProcessOrNot,
   sd::LongType* allocationBuffer,
   void* reductionBuffer,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets) {

 simpleIndexReduceGeneric<X,Z>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         opNum,
         dx,
         xShapeInfo,
         xRank,
         extraParams,
         result,
         zShapeInfo,
         0,    // we pass 0 for zRank to match original logic
         nullptr,
         0,
         1,
         allocationBuffer,
         reductionBuffer,
         tadOnlyShapeInfo,
         tadOffsets);

 sd::DebugHelper::checkErrorCode(
     stream, "executeIndexReduceScalar(...) failed");
}

template <typename X, typename Z>
SD_HOST void IndexReduce<X,Z>::executeIndexReduce(
   dim3 launchDims,
   cudaStream_t* stream,
   const int opNum,
   void const* dx,
   sd::LongType const* xShapeInfo,
   sd::LongType xRank,
   void* extraParams,
   void* result,
   sd::LongType const* zShapeInfo,
   sd::LongType zRank,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   int postProcessOrNot,
   sd::LongType* allocationBuffer,
   void* reductionBuffer,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets) {

 simpleIndexReduceGeneric<X,Z>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         opNum,
         dx,
         xShapeInfo,
         xRank,
         extraParams,
         result,
         zShapeInfo,
         zRank,
         dimension,
         dimensionLength,
         postProcessOrNot,
         allocationBuffer,
         reductionBuffer,
         tadOnlyShapeInfo,
         tadOffsets);

 sd::DebugHelper::checkErrorCode(stream, "executeIndexReduce(...) failed");
}

// This is the un-specialized struct placeholder
template <typename T>
struct SharedIndexValue {
 SD_DEVICE T* getPointer() {
   // We'll never instantiate an un-specialized type
   extern SD_DEVICE void error(void);
   error();
   return nullptr;
 }
};

// We create specializations for float/double etc
template <>
struct SharedIndexValue<float> {
 SD_DEVICE IndexValue<float>* getPointer() {
   extern __shared__ IndexValue<float> s_int2[];
   return s_int2;
 }
};

template <>
struct SharedIndexValue<double> {
 SD_DEVICE IndexValue<double>* getPointer() {
   extern __shared__ IndexValue<double> s_int6[];
   return s_int6;
 }
};

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void IndexReduce<X,Z>::aggregatePartials(
   IndexValue<X>* sPartials,
   sd::LongType tid,
   sd::LongType numElements,
   void* vextraParams) {

 auto extraParams = static_cast<X*>(vextraParams);

 sd::LongType floorPow2 = static_cast<sd::LongType>(blockDim.x);
 if (floorPow2 & (floorPow2 - 1)) {
   while (floorPow2 & (floorPow2 - 1)) {
     floorPow2 &= (floorPow2 - 1);
   }
   if (tid >= floorPow2) {
     IndexValue<X> prev = sPartials[tid - floorPow2];
     IndexValue<X> curr = sPartials[tid];
     sPartials[tid - floorPow2] = OpType::update(prev, curr, extraParams);
   }
   __syncthreads();
 }

 for (sd::LongType active = floorPow2 >> 1; active; active >>= 1) {
   if (tid < active && tid + active < numElements) {
     IndexValue<X> curr = sPartials[tid];
     IndexValue<X> next = sPartials[tid + active];
     sPartials[tid] = OpType::update(curr, next, extraParams);
   }
   __syncthreads();
 }
}

template <typename X, typename Y>
SD_DEVICE void IndexReduce<X,Y>::transform(
   int opNum,
   void const* x,
   sd::LongType const* xShapeInfo,
   void* extraParams,
   void* result,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   int postProcessOrNot,
   sd::LongType* allocationBuffer,
   void* reductionBuffer,
   sd::LongType const* tadShapeInfo,
   sd::LongType const* tadOffsets) {

 DISPATCH_BY_OPNUM_TT(
     transform,
     PARAMS(x, xShapeInfo, extraParams, result, zShapeInfo,
            dimension, dimensionLength, postProcessOrNot,
            allocationBuffer, reductionBuffer,
            tadShapeInfo, tadOffsets),
     INDEX_REDUCE_OPS);
}

template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void IndexReduce<X,Z>::transform(
   void const* vdx,
   sd::LongType const* xShapeInfo,
   void* vextraParams,
   void* vz,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   int postProcessOrNot,
   sd::LongType* allocationBuffer,
   void* vreductionBuffer,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets) {

 auto dx = reinterpret_cast<const X*>(vdx);
 auto z  = reinterpret_cast<Z*>(vz);
 auto extraParams    = static_cast<X*>(vextraParams);
 auto reductionBuff  = static_cast<unsigned int*>(vreductionBuffer);
 auto xOrder         = shape::order(xShapeInfo);
 sd::LongType tid    = blockIdx.x * blockDim.x + threadIdx.x;

 // We'll place partial sums in shared memory
 __shared__ IndexValue<X> sPartials[SD_CUDA_BLOCK_SIZE];

 // Initialize starting value for this thread
 sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

 // We'll load length into shared memory
 __shared__ sd::LongType xLength;
 __shared__ sd::LongType zLen;
 __shared__ bool resultScalar;

 // Initialize shared variables from thread 0
 if (threadIdx.x == 0) {
   xLength     = shape::length(xShapeInfo);
   if (zShapeInfo != nullptr) {
     zLen      = shape::length(zShapeInfo);
   } else {
     zLen      = 1;
   }
   resultScalar = (zLen == 1);
 }
 __syncthreads();

 // Local reduction variable for this thread
 IndexValue<X> reduction = OpType::startingIndexValue(dx);

 sd::LongType threadIdxX = static_cast<sd::LongType>(threadIdx.x);
 sd::LongType blockDimX = static_cast<sd::LongType>(blockDim.x);
 sd::LongType blockIdxX = static_cast<sd::LongType>(blockIdx.x);
 sd::LongType gridDimX = static_cast<sd::LongType>(gridDim.x);

 // Handle different reduction scenarios
 if (!resultScalar) {
   // TAD (Tensor Along Dimension) based reduction
   __shared__ sd::LongType tadLength;
   __shared__ sd::LongType numTads;

   if (threadIdx.x == 0) {
     tadLength = shape::length(tadOnlyShapeInfo);
     numTads = shape::length(xShapeInfo) / tadLength;
   }
   __syncthreads();

   // Process each TAD
   for (sd::LongType r = blockIdxX; r < numTads; r += gridDimX) {
     auto tadOffsetForBlock = tadOffsets[r];
     sPartials[threadIdxX] = OpType::startingIndexValue(dx);

     // Reduce within this TAD using index-to-coordinate conversion
     for (sd::LongType i = threadIdxX; i < tadLength; i += blockDimX) {
       auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo);
       IndexValue<X> comp{dx[xOffset], i};
       sPartials[threadIdxX] = OpType::update(sPartials[threadIdxX], comp, extraParams);
     }

     __syncthreads();
     aggregatePartials<OpType>(sPartials, threadIdxX,
                               sd::math::sd_min<sd::LongType,sd::LongType>(blockDimX, tadLength),
                               extraParams);

     __syncthreads();
     if (threadIdxX == 0) {
       z[r] = static_cast<Z>(sPartials[threadIdxX].index);
     }
     __syncthreads();
   }
 } else {
   // Scalar reduction - process entire array
   auto n = shape::length(xShapeInfo);

   // Always use coordinate-based indexing (no element-wise stride optimization)
   for (sd::LongType i = tid; i < n; i += (gridDimX * blockDimX)) {
     // Use shape::getIndexOffset for proper index-to-coordinate conversion
     auto xOffset = shape::getIndexOffset(i, xShapeInfo);
     IndexValue<X> comp{dx[xOffset], i};
     reduction = OpType::update(reduction, comp, extraParams);
   }

   // Store thread's partial result
   sPartials[threadIdxX] = reduction;
   __syncthreads();

   // Aggregate partial results within block
   aggregatePartials<OpType>(sPartials, threadIdxX,
                             sd::math::sd_min<sd::LongType,sd::LongType>(blockDimX, n),
                             extraParams);

   // Handle multi-block reductions
   if (gridDimX > 1) {
     __shared__ bool amLast;
     unsigned int* unsignedSharedMemory = reductionBuff;

     // Store block result
     if (threadIdx.x == 0) {
       reductionBuff[blockIdx.x] = sPartials[threadIdx.x].index;
     }

     __threadfence();
     __syncthreads();

     // Check if this is the last block to finish
     if (threadIdx.x == 0) {
       unsigned int ticket = atomicInc(&unsignedSharedMemory[16384], gridDim.x);
       amLast = (ticket == gridDim.x - 1);
     }
     __syncthreads();

     // Final reduction across blocks
     if (amLast) {
       sPartials[threadIdx.x] = OpType::startingIndexValue(dx);
       for (sd::LongType i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
         IndexValue<X> comp{static_cast<X>(0), reductionBuff[i]};
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
     // Single block - store result directly
     if (threadIdx.x == 0) {
       z[0] = static_cast<Z>(sPartials[threadIdx.x].index);
     }
   }
 }
}

}  // namespace indexreduce
}  // namespace functions