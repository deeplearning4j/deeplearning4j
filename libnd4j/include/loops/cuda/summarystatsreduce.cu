/******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>
#include <loops/summarystatsreduce.h>
#include <ops/specials_cuda.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/float16.h>
#include <types/types.h>

    using namespace simdOps;

namespace functions {
namespace summarystats {

/**
 * The new kernel that caches shape info for x/tad usage in shared memory,
 * then calls SummaryStatsReduce transform, performing index->coords->offset
 * transformations with the cached pointers.
 */
template <typename X, typename Z>
__global__ void summaryStatsReduceCachedKernel(
    int op,
    void const* dx,
    sd::LongType const* xShapeInfo,
    sd::LongType xRank,
    void* extraParams,
    void* z,
    sd::LongType const* zShapeInfo,
    sd::LongType zRank,
    sd::LongType* dimension,
    long long int dimensionLength,
    int postProcessOrNot,
    bool biasCorrected,
    sd::LongType* allocationBuffer,
    void* reductionBuffer,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets)
{
  // We'll do the shape-caching logic inside the transform method.
  // For instance, we call SummaryStatsReduce::transform below,
  // which previously repeated shapeOf/strideOf calls frequently.
  // We'll keep the logic as is, but the transform method is the key place
  // that does the index->coords->offset transformations with caching.

  SummaryStatsReduce<X,Z>::transform(
      op,
      dx,
      xShapeInfo,
      extraParams,
      z,
      zShapeInfo,
      dimension,
      dimensionLength,
      biasCorrected,
      allocationBuffer,
      reductionBuffer,
      tadOnlyShapeInfo,
      tadOffsets
  );
}

///////////////////////////////////////////////////////////////////
// We rewrite summaryStatsReduceT() to directly launch the new kernel.
template <typename X, typename Z>
SD_DEVICE void summaryStatsReduceT(
    int op,
    void const* dx,
    sd::LongType const* xShapeInfo,
    sd::LongType xRank,
    void* extraParams,
    void* z,
    sd::LongType const* zShapeInfo,
    sd::LongType zRank,
    sd::LongType* dimension,
    long long int dimensionLength,
    int postProcessOrNot,
    bool biasCorrected,
    sd::LongType* allocationBuffer,
    void* reductionBuffer,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets)
{
  // This is a device function in the original snippet,
  // but it's basically a forwarder to transform. We'll keep it as a no-op or direct call.
  // The logic is in the kernel now anyway, so no changes here.
  // We'll call transform directly if needed.
  SummaryStatsReduce<X,Z>::transform(op,
                                      dx,
                                      xShapeInfo,
                                      extraParams,
                                      z,
                                      zShapeInfo,
                                      dimension,
                                      dimensionLength,
                                      biasCorrected,
                                      allocationBuffer,
                                      reductionBuffer,
                                      tadOnlyShapeInfo,
                                      tadOffsets);
}

///////////////////////////////////////////////////////////////////
// We'll keep the partial aggregator the same, but note that
// the aggregator uses shared memory sPartials for partial merges.
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void SummaryStatsReduce<X, Z>::aggregatePartials(
    SummaryStatsData<X>* sPartials,
    sd::LongType tid,
    sd::LongType numElements,
    void* vextraParams)
{
  auto extraParams = static_cast<Z*>(vextraParams);
  sd::LongType floorPow2 = blockDim.x;

  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) {
      floorPow2 &= floorPow2 - 1;
    }
    if (tid >= floorPow2) {
      auto prev = sPartials[tid - floorPow2];
      auto curr = sPartials[tid];
      sPartials[tid - floorPow2] = update(prev, curr, extraParams);
    }
    __syncthreads();
  }

  for (sd::LongType activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads && tid + activeThreads < numElements) {
      auto curr = sPartials[tid];
      auto next = sPartials[tid + activeThreads];
      sPartials[tid] = update(curr, next, extraParams);
    }
    __syncthreads();
  }
}
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void SummaryStatsReduce<X,Z>::transform(
    int op,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType const* zShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    bool biasCorrected,
    sd::LongType* allocationBuffer,
    void* vreductionBuffer,
    sd::LongType const* tadOnlyShapeInfo,
    sd::LongType const* tadOffsets)
{
  auto dx = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);
  auto extraParams = reinterpret_cast<Z*>(vextraParams);
  auto reductionBuf = reinterpret_cast<Z*>(vreductionBuffer);

  // Shape data caching in shared memory
  __shared__ sd::LongType xLen;
  __shared__ sd::LongType zLen;
  __shared__ int xRank, zRank;
  __shared__ const sd::LongType* xShape;
  __shared__ const sd::LongType* xStride;
  __shared__ const sd::LongType* zShape;
  __shared__ const sd::LongType* zStride;

  // TAD shape data caching
  __shared__ sd::LongType tadLen;
  __shared__ sd::LongType numTads;
  __shared__ int tadRank;
  __shared__ const sd::LongType* tadShape;
  __shared__ const sd::LongType* tadStride;

  // Partial results in shared memory
  __shared__ SummaryStatsData<X> sPartials[SD_CUDA_BLOCK_SIZE];
  __shared__ int isScalarResult;

  if (threadIdx.x == 0) {
    // Cache X shape data
    xLen = shape::length(xShapeInfo);
    xRank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);

    // Cache Z shape data if available
    if (zShapeInfo != nullptr) {
      zLen = shape::length(zShapeInfo);
      zRank = shape::rank(zShapeInfo);
      zShape = shape::shapeOf(zShapeInfo);
      zStride = shape::stride(zShapeInfo);
    } else {
      zLen = 1;
      zRank = 0;
      zShape = nullptr;
      zStride = nullptr;
    }

    // Cache TAD shape data
    if (tadOnlyShapeInfo != nullptr) {
      tadLen = shape::length(tadOnlyShapeInfo);
      tadRank = shape::rank(tadOnlyShapeInfo);
      tadShape = shape::shapeOf(tadOnlyShapeInfo);
      tadStride = shape::stride(tadOnlyShapeInfo);
      numTads = xLen / tadLen;
    }

    isScalarResult = (zLen == 1) ? 1 : 0;
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize starting values
  Z startVal = startingValue(dx);
  SummaryStatsData<X> localVal;
  localVal.initWithValue(startVal);
  localVal.n = 0;
  sPartials[threadIdx.x] = localVal;
  __syncthreads();

  // Non-scalar result case
  if (!isScalarResult) {
    for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
      auto tadOffset = tadOffsets[r];

      SummaryStatsData<X> initVal;
      initVal.initWithValue(startVal);
      initVal.n = 0;
      sPartials[threadIdx.x] = initVal;
      __syncthreads();

      // Process TAD
      for (sd::LongType i = threadIdx.x; i < tadLen; i += blockDim.x) {
        sd::LongType coords[SD_MAX_RANK];
        sd::LongType offset;
        INDEX2COORDS(i, tadRank, tadShape, coords);
        COORDS2INDEX(tadRank, tadStride, coords, offset);

        auto xOffsetFinal = tadOffset + offset;
        SummaryStatsData<X> xVal;
        xVal.initWithValue(dx[xOffsetFinal]);

        sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(xVal, extraParams), extraParams);
      }
      __syncthreads();

      // Reduce partials
      aggregatePartials<OpType>(sPartials, threadIdx.x, sd::math::sd_min<int>(blockDim.x, tadLen), extraParams);
      __syncthreads();

      if (threadIdx.x == 0) {
        z[r] = OpType::getValue(/*postProcessOrNot =*/1, sPartials[0]);
      }
      __syncthreads();
    }
  }
  // Scalar result case
  else {
    SummaryStatsData<X> accumVal;
    accumVal.initWithValue(0.0);
    accumVal.n = 0;

    for (sd::LongType i = tid; i < xLen; i += blockDim.x * gridDim.x) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType offset;
      INDEX2COORDS(i, xRank, xShape, coords);
      COORDS2INDEX(xRank, xStride, coords, offset);

      SummaryStatsData<X> xVal;
      xVal.initWithValue(dx[offset]);

      accumVal = update(accumVal, OpType::op(xVal, extraParams), extraParams);
    }
    sPartials[threadIdx.x] = accumVal;
    __syncthreads();

    // Reduce partials
    aggregatePartials<OpType>(sPartials, threadIdx.x, blockDim.x, extraParams);
    __syncthreads();

    // Multiple blocks case
    if (gridDim.x > 1) {
      __shared__ bool amLast;
      auto tc = reinterpret_cast<unsigned int*>(reductionBuf);

      if (threadIdx.x == 0) {
        auto pBuf = reinterpret_cast<SummaryStatsData<X>*>(reductionBuf);
        pBuf[blockIdx.x] = sPartials[0];
      }
      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0) {
        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
        amLast = (ticket == gridDim.x - 1);
      }
      __syncthreads();

      if (amLast) {
        tc[16384] = 0;
        auto pBuf = reinterpret_cast<SummaryStatsData<X>*>(reductionBuf);

        SummaryStatsData<X> finalVal;
        finalVal.initWithValue(startVal);
        finalVal.n = 0;
        sPartials[threadIdx.x] = finalVal;

        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
          sPartials[threadIdx.x] = update(sPartials[threadIdx.x], pBuf[i], extraParams);
        }
        __syncthreads();

        aggregatePartials<OpType>(sPartials, threadIdx.x, gridDim.x, extraParams);
        __syncthreads();

        if (threadIdx.x == 0) {
          z[0] = OpType::getValue(/*postProcessOrNot =*/1, sPartials[0]);
        }
      }
    }
    // Single block case
    else {
      if (tid == 0) {
        auto tc = reinterpret_cast<unsigned int*>(reductionBuf);
        tc[16384] = 0;
        z[0] = OpType::getValue(/*postProcessOrNot =*/1, sPartials[0]);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////
// The host code that calls the new kernel
template <typename X, typename Z>
SD_HOST void SummaryStatsReduce<X,Z>::execSummaryStatsReduceScalar(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    sd::LongType const* hxShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType const* zShapeInfo,
    sd::LongType const* hzShapeInfo,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    bool biasCorrected,
    void* reductionBuffer)
{
  if (sd::Environment::getInstance().isDebugAndVerbose()) {
    printf("SummStatsReduce scalar: opNum=%d\n", opNum);
  }

  summaryStatsReduceCachedKernel<X,Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      opNum,
      vx,
      xShapeInfo,
      shape::rank(hxShapeInfo),
      vextraParams,
      vz,
      zShapeInfo,
      shape::rank(hzShapeInfo),
      nullptr,          // dimension
      1L,               // dimensionLength
      1,                // postProcessOrNot
      biasCorrected,
      nullptr,
      reductionBuffer,
      tadShapeInfo,
      tadOffsets
  );
  sd::DebugHelper::checkErrorCode(stream, "execSummaryStatsReduceScalar(...) failed");
}

template <typename X, typename Z>
SD_HOST void SummaryStatsReduce<X,Z>::execSummaryStatsReduce(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    sd::LongType const* hxShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType const* zShapeInfo,
    sd::LongType const* hzShapeInfo,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    bool biasCorrected,
    void* reductionBuffer)
{
  if (sd::Environment::getInstance().isDebugAndVerbose()) {
    printf("SummStatsReduce exec, no dimension param: opNum=%d\n", opNum);
  }

  summaryStatsReduceCachedKernel<X,Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      opNum,
      vx,
      xShapeInfo,
      shape::rank(hxShapeInfo),
      vextraParams,
      vz,
      zShapeInfo,
      shape::rank(hzShapeInfo),
      nullptr,
      1L,           // dimensionLength
      1,            // postProcessOrNot
      biasCorrected,
      nullptr,
      reductionBuffer,
      tadShapeInfo,
      tadOffsets
  );
  DEBUG_KERNEL(stream, opNum);
}

template <typename X, typename Z>
SD_HOST void SummaryStatsReduce<X,Z>::execSummaryStatsReduce(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    sd::LongType const* hxShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType const* zShapeInfo,
    sd::LongType const* hzShapeInfo,
    sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    bool biasCorrected,
    void* reductionBuffer)
{
  if (sd::Environment::getInstance().isDebugAndVerbose()) {
    printf("SummStatsReduce exec w/ dimension param: opNum=%d\n", opNum);
  }

  summaryStatsReduceCachedKernel<X,Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      opNum,
      vx,
      xShapeInfo,
      shape::rank(hxShapeInfo),
      vextraParams,
      vz,
      zShapeInfo,
      shape::rank(hzShapeInfo),
      dimension,
      dimensionLength,
      1, // postProcessOrNot
      biasCorrected,
      nullptr,
      reductionBuffer,
      tadShapeInfo,
      tadOffsets
  );
  sd::DebugHelper::checkErrorCode(stream, "SummaryStatsReduce execSummaryStatsReduce(...) failed");
}

BUILD_DOUBLE_TEMPLATE(template class SummaryStatsReduce, , SD_COMMON_TYPES, SD_FLOAT_TYPES);

}  // namespace summarystats
}  // namespace functions
