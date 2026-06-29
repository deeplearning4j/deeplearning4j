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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/top_k.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"
#include <system/selective_rendering.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL static void inTopKCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                  const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                  const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                  const LongType k) {
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<bool*>(vz);

  // Shared memory for caching shape information and per-thread reduction counts.
  // Dynamic region (extern __shared__) holds one LongType slot per thread for the
  // parallel-reduction counter.  Named __shared__ vars cache shape metadata.
  __shared__ LongType shared_yRank;
  __shared__ const LongType* shared_yShape;
  __shared__ const LongType* shared_yStride;

  __shared__ LongType shared_zRank;
  __shared__ const LongType* shared_zShape;
  __shared__ const LongType* shared_zStride;

  __shared__ LongType shared_xTadRank;
  __shared__ const LongType* shared_xTadShape;
  __shared__ const LongType* shared_xTadStride;

  __shared__ X elemToCompare;
  __shared__ LongType xTadLen;

  // ONE shared coords array reused for all per-thread-0 coordinate lookups,
  // avoiding multiple SD_MAX_RANK-sized stack arrays that overflow CUDA per-thread
  // local memory (32 × 8 bytes = 256 bytes each; four arrays = 1024 bytes ≥ limit).
  __shared__ LongType shared_coords[SD_MAX_RANK];

  // Initialize shared metadata (thread 0 only)
  if (threadIdx.x == 0) {
    shared_yRank = shape::rank(yShapeInfo);
    shared_zRank = shape::rank(zShapeInfo);
    shared_xTadRank = shape::rank(xTadShapeInfo);

    shared_yShape  = shape::shapeOf(yShapeInfo);
    shared_zShape  = shape::shapeOf(zShapeInfo);
    shared_xTadShape = shape::shapeOf(xTadShapeInfo);

    shared_yStride  = shape::stride(yShapeInfo);
    shared_zStride  = shape::stride(zShapeInfo);
    shared_xTadStride = shape::stride(xTadShapeInfo);

    xTadLen = shape::length(xTadShapeInfo);

    // Compute target index for this block: y[blockIdx.x].
    // Targets (y) is always rank-1 for in_top_k; compute yOffset directly from the
    // stride scalar.  The target column is stored as targetIdx (not 'idx') to avoid
    // a name collision with the internal 'idx' variable inside the INDEX2COORDS macro.
    // If the outer variable were named 'idx', the macro's `sd::LongType idx = (idx);`
    // expansion would self-initialize from an uninitialized inner 'idx' (undefined
    // behaviour), causing shared_coords[0] = 0 for every block → elemToCompare always
    // reads xTad[0] → count is always off → all outputs wrong.
    LongType yOffset = static_cast<LongType>(blockIdx.x) * shared_yStride[0];
    LongType targetIdx = y[yOffset];

    // Find element to compare: xTad[targetIdx]
    const X* xTadPtr = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];
    LongType xOffset;
    INDEX2COORDS(targetIdx, shared_xTadRank, shared_xTadShape, shared_coords);
    COORDS2INDEX(shared_xTadRank, shared_xTadStride, shared_coords, xOffset);
    elemToCompare = xTadPtr[xOffset];
  }

  __syncthreads();

  // Per-thread reduction counter in dynamic shared memory (one slot per thread).
  extern __shared__ LongType sharedMem[];
  sharedMem[threadIdx.x] = 0;
  __syncthreads();

  // Count how many elements in this TAD strictly exceed elemToCompare.
  // Each thread handles a strided subset; coords are kept on the (small) stack
  // because only one array is needed at a time here.
  const X* xTad = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];
  for (LongType i = threadIdx.x; i < xTadLen; i += blockDim.x) {
    LongType coords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(i, shared_xTadRank, shared_xTadShape, coords);
    COORDS2INDEX(shared_xTadRank, shared_xTadStride, coords, xOffset);
    if (elemToCompare < xTad[xOffset]) {
      sharedMem[threadIdx.x]++;
    }
  }

  __syncthreads();

  // Parallel reduction: sum counts across all threads in the block.
  for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
    if (threadIdx.x < activeThreads) {
      sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
    }
    __syncthreads();
  }

  // Thread 0 writes the boolean result: true if fewer than k elements exceed the target.
  // z (output) is always rank-1 for in_top_k; compute zOffset directly to avoid
  // any residual shared_coords aliasing.
  if (threadIdx.x == 0) {
    LongType zOffset = static_cast<LongType>(blockIdx.x) * shared_zStride[0];
    z[zOffset] = (sharedMem[0] < k);
  }
}
//////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void inTopKCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                               const void* vy, const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                               const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                               const LongType k) {
  inTopKCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                           zShapeInfo, xTadShapeInfo, xTadOffsets, k);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "inTopKCudaLauncher failed");

}

///////////////////////////////////////////////////////////////////
Status inTopKFunctor(LaunchContext* context, NDArray* predictions, NDArray* targets,
                         NDArray* output, const LongType k) {
  PointersManager manager(context, "in_top_k");

  // TAD over the LAST dimension (the classes axis) → one TAD per batch row, so numberOfTads ==
  // batch == targets.length(). This matches topKFunctor's {rankOf()-1} usage below. ({0} would
  // create one scalar TAD per element — the OPPOSITE — overrunning the targets buffer.)
  // rankOf()-1 is a runtime LongType expr (not the literal {0}), so it resolves unambiguously to
  // the single-dimension tadForDimensions(LongType*, LongType) overload.
  const auto packX = ConstantTadHelper::getInstance().tadForDimensions(predictions->shapeInfo(),
                                                                       static_cast<sd::LongType>(predictions->rankOf() - 1));

  dim3 topkDims2 = topkDims(packX->numberOfTads());
  const auto xType = predictions->dataType();
  const auto yType = targets->dataType();

  NDArray::prepareSpecialUse({output}, {predictions, targets});
  BUILD_DOUBLE_SELECTOR(
      xType, yType, inTopKCudaLauncher,
      (topkDims2.x, topkDims2.y, topkDims2.z, context->getCudaStream(), predictions->specialBuffer(),
          predictions->specialShapeInfo(), targets->specialBuffer(), targets->specialShapeInfo(), output->specialBuffer(),
          output->specialShapeInfo(), packX->specialShapeInfo(), packX->specialOffsets(), k),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {predictions, targets});

  manager.synchronize();

  return Status::OK;
}

template <typename X, typename Y>
static SD_KERNEL void topValuesMover(void const* vx, LongType const* xTadShapeInfo, LongType const* xTadOffsets,
                                     void const* vi, LongType const* iTadShapeInfo, LongType const* iTadOffsets,
                                     void* vz, LongType const* zTadShapeInfo, LongType const* zTadOffsets,
                                     LongType tadLength, int numTads, int k) {
  // Cache shape information in shared memory
  __shared__ int xRank, iRank, zRank;
  __shared__ LongType *xShape, *iShape, *zShape;
  __shared__ LongType *xStride, *iStride, *zStride;

  if (threadIdx.x == 0) {
    // Cache ranks
    xRank = shape::rank(xTadShapeInfo);
    iRank = shape::rank(iTadShapeInfo);
    zRank = shape::rank(zTadShapeInfo);

    // Cache shapes
    xShape = shape::shapeOf(xTadShapeInfo);
    iShape = shape::shapeOf(iTadShapeInfo);
    zShape = shape::shapeOf(zTadShapeInfo);

    // Cache strides
    xStride = shape::stride(xTadShapeInfo);
    iStride = shape::stride(iTadShapeInfo);
    zStride = shape::stride(zTadShapeInfo);
  }
  __syncthreads();

  for (int t = blockIdx.x; t < numTads; t += gridDim.x) {
    auto x = reinterpret_cast<X const*>(vx) + xTadOffsets[t];
    auto i = reinterpret_cast<Y const*>(vi) + iTadOffsets[t];
    auto z = reinterpret_cast<X*>(vz) + zTadOffsets[t];

    LongType iCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];
    LongType xCoords[SD_MAX_RANK];
    LongType iOffset;
    LongType zOffset;
    LongType xOffset;

    for (int e = threadIdx.x; e < k; e += blockDim.x) {
      INDEX2COORDS(e, iRank, iShape, iCoords);
      COORDS2INDEX(iRank, iStride, iCoords, iOffset);
      auto srcIdx = i[iOffset];

      INDEX2COORDS(e, zRank, zShape, zCoords);
      COORDS2INDEX(zRank, zStride, zCoords, zOffset);

      INDEX2COORDS(srcIdx, xRank, xShape, xCoords);
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);

      z[zOffset] = x[xOffset];
    }
  }
}

template <typename X, typename Y>
static SD_KERNEL void indicesAlongDimension(void const* vx, LongType const* xTadShapeInfo, LongType const* xTadOffsets, void* vi, LongType const* iTadShapeInfo, LongType const* iTadOffsets,
                                            void* vz, LongType const* zTadShapeInfo, LongType const* zTadOffsets,
                                            LongType tadLength, int numTads, int k,
                                            int scanWidth, bool needSort) {
  extern __shared__ char _shmem[];

  X* tempValues = reinterpret_cast<X*>(_shmem) + threadIdx.x * scanWidth;
  Y* tempIndices =
      reinterpret_cast<Y*>(reinterpret_cast<X*>(_shmem) + blockDim.x * scanWidth) + threadIdx.x * scanWidth;

  // Cache shape information in shared memory
  __shared__ int xRank, iRank, zRank;
  __shared__ LongType *xShape, *iShape, *zShape;
  __shared__ LongType *xStride, *iStride, *zStride;
  __shared__ X localMaximum;

  if (threadIdx.x == 0) {
    localMaximum = -DataTypeUtils::max<X>();

    // Cache ranks
    xRank = shape::rank(xTadShapeInfo);
    iRank = shape::rank(iTadShapeInfo);
    zRank = shape::rank(zTadShapeInfo);

    // Cache shapes
    xShape = shape::shapeOf(xTadShapeInfo);
    iShape = shape::shapeOf(iTadShapeInfo);
    zShape = shape::shapeOf(zTadShapeInfo);

    // Cache strides
    xStride = shape::stride(xTadShapeInfo);
    iStride = shape::stride(iTadShapeInfo);
    zStride = shape::stride(zTadShapeInfo);
  }
  __syncthreads();

  for (int t = blockIdx.x; t < numTads; t += gridDim.x) {
    auto x = reinterpret_cast<X const*>(vx) + xTadOffsets[t];
    auto i = reinterpret_cast<Y*>(vi) + iTadOffsets[t];
    auto z = reinterpret_cast<X*>(vz) + zTadOffsets[t];

    // Reset localMaximum for each new TAD
    if (threadIdx.x == 0) {
      localMaximum = DataTypeUtils::max<X>();
    }
    __syncthreads();

    // we'll do multiple reads here - find top k values one at a time
    for (int kIdx = 0; kIdx < k; kIdx++) {
      // resetting temporary storage for this thread
      tempValues[0] = -DataTypeUtils::max<X>();
      tempIndices[0] = -1;

      // Each thread finds max value in its portion of the array
      for (int e = threadIdx.x; e < tadLength; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType xOffset;
        INDEX2COORDS(e, xRank, xShape, xCoords);
        COORDS2INDEX(xRank, xStride, xCoords, xOffset);
        auto value = x[xOffset];

        // Only consider values smaller than current threshold (localMaximum)
        // For the first iteration (kIdx==0), localMaximum is max<X>(), so all values pass
        if (value < localMaximum && value > tempValues[0]) {
          tempValues[0] = value;
          tempIndices[0] = e;
        }
      }
      __syncthreads();

      // Parallel reduction to find global max across all threads
      // Use shared memory layout: each thread's data is at index threadIdx.x * scanWidth
      for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
        if (threadIdx.x < activeThreads) {
          X* otherValues = reinterpret_cast<X*>(_shmem) + (threadIdx.x + activeThreads) * scanWidth;
          Y* otherIndices = reinterpret_cast<Y*>(reinterpret_cast<X*>(_shmem) + blockDim.x * scanWidth) + (threadIdx.x + activeThreads) * scanWidth;

          if (otherValues[0] > tempValues[0]) {
            tempValues[0] = otherValues[0];
            tempIndices[0] = otherIndices[0];
          }
        }
        __syncthreads();
      }

      // Thread 0 writes the result for this k index
      if (threadIdx.x == 0) {
        // Update localMaximum to be the value we just found (for next iteration)
        localMaximum = tempValues[0];

        LongType zCoords[SD_MAX_RANK];
        LongType zOffset;
        INDEX2COORDS(kIdx, zRank, zShape, zCoords);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);
        z[zOffset] = tempValues[0];

        LongType iCoords[SD_MAX_RANK];
        LongType iOffset;
        INDEX2COORDS(kIdx, iRank, iShape, iCoords);
        COORDS2INDEX(iRank, iStride, iCoords, iOffset);
        i[iOffset] = tempIndices[0];
      }
      __syncthreads();
    }

    __syncthreads();
    if (!needSort) {
      // if we don't need sort, we need to return values based on their indices (ascending)
      for (int m = 0; m < k; m++) {
        if (m % 2 == 0) {
          for (int tid = threadIdx.x; tid < k; tid += blockDim.x) {
            auto top = 2 * tid + 1;
            if (top < k) {
              LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
              LongType t0Offset, t1Offset;

              INDEX2COORDS(top - 1, iRank, iShape, t0Coords);
              COORDS2INDEX(iRank, iStride, t0Coords, t0Offset);
              INDEX2COORDS(top, iRank, iShape, t1Coords);
              COORDS2INDEX(iRank, iStride, t1Coords, t1Offset);

              if (i[t0Offset] > i[t1Offset]) {
                // swap indices first
                Y di0 = i[t0Offset];
                i[t0Offset] = i[t1Offset];
                i[t1Offset] = di0;

                // swap values next
                LongType zT0Coords[SD_MAX_RANK], zT1Coords[SD_MAX_RANK];
                LongType zT0Offset, zT1Offset;

                INDEX2COORDS(top - 1, zRank, zShape, zT0Coords);
                COORDS2INDEX(zRank, zStride, zT0Coords, zT0Offset);
                INDEX2COORDS(top, zRank, zShape, zT1Coords);
                COORDS2INDEX(zRank, zStride, zT1Coords, zT1Offset);

                X dz0 = z[zT0Offset];
                z[zT0Offset] = z[zT1Offset];
                z[zT1Offset] = dz0;
              }
            }
          }
        } else {
          for (int tid = threadIdx.x; tid < k; tid += blockDim.x) {
            auto top = 2 * tid + 2;
            if (top < k) {
              LongType t0Coords[SD_MAX_RANK], t1Coords[SD_MAX_RANK];
              LongType t0Offset, t1Offset;

              INDEX2COORDS(top - 1, iRank, iShape, t0Coords);
              COORDS2INDEX(iRank, iStride, t0Coords, t0Offset);
              INDEX2COORDS(top, iRank, iShape, t1Coords);
              COORDS2INDEX(iRank, iStride, t1Coords, t1Offset);

              if (i[t0Offset] > i[t1Offset]) {
                // swap indices first
                Y di0 = i[t0Offset];
                i[t0Offset] = i[t1Offset];
                i[t1Offset] = di0;

                // swap values next
                LongType zT0Coords[SD_MAX_RANK], zT1Coords[SD_MAX_RANK];
                LongType zT0Offset, zT1Offset;

                INDEX2COORDS(top - 1, zRank, zShape, zT0Coords);
                COORDS2INDEX(zRank, zStride, zT0Coords, zT0Offset);
                INDEX2COORDS(top, zRank, zShape, zT1Coords);
                COORDS2INDEX(zRank, zStride, zT1Coords, zT1Offset);

                X dz0 = z[zT0Offset];
                z[zT0Offset] = z[zT1Offset];
                z[zT1Offset] = dz0;
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
}
template <typename X, typename Y>
static Status topKFunctor_(LaunchContext* context, NDArray* input, NDArray* values, NDArray* indices,
                           const LongType k, bool needSort) {
  // For 1D arrays, tadForDimensions({0}) creates N scalar TADs instead of 1 TAD of length N
  // We need to handle 1D arrays specially by using the array's own shape info

  LongType tadLength;
  LongType numTads;
  const LongType* xTadShapeInfo;
  const LongType* xTadOffsets;
  const LongType* iTadShapeInfo;
  const LongType* iTadOffsets;
  const LongType* zTadShapeInfo;
  const LongType* zTadOffsets;

  // Device memory for zero offset (needed for async kernel execution)
  LongType* deviceZeroOffset = nullptr;
  int topkDevId = context->getDeviceID();

  std::shared_ptr<TadPack> packX, packI, packZ;

  if (input->rankOf() == 1) {
    // For 1D input, treat the entire array as one TAD
    // Use the array's own shape info directly
    tadLength = input->lengthOf();
    numTads = 1;

    // Use the arrays' own shape info (they represent the TAD shape for 1D case)
    xTadShapeInfo = input->specialShapeInfo();
    iTadShapeInfo = indices->specialShapeInfo();
    zTadShapeInfo = values->specialShapeInfo();

    // Allocate device memory for the zero offset
    deviceZeroOffset = reinterpret_cast<LongType*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(LongType), topkDevId, *context->getCudaStream()));
    if (deviceZeroOffset == nullptr) THROW_EXCEPTION("Cannot allocate memory for top_k zero offset");
    cudaMemsetAsync(deviceZeroOffset, 0, sizeof(LongType), *context->getCudaStream());

    // Single TAD starting at offset 0
    xTadOffsets = deviceZeroOffset;
    iTadOffsets = deviceZeroOffset;
    zTadOffsets = deviceZeroOffset;
  } else {
    // For multi-dimensional arrays, use standard TAD along last dimension
    packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {input->rankOf() - 1});
    packI = ConstantTadHelper::getInstance().tadForDimensions(indices->shapeInfo(), {input->rankOf() - 1});
    packZ = ConstantTadHelper::getInstance().tadForDimensions(values->shapeInfo(), {input->rankOf() - 1});

    tadLength = shape::length(packX->primaryShapeInfo());
    numTads = packX->numberOfTads();

    xTadShapeInfo = packX->platformShapeInfo();
    xTadOffsets = packX->platformOffsets();
    iTadShapeInfo = packI->platformShapeInfo();
    iTadOffsets = packI->platformOffsets();
    zTadShapeInfo = packZ->platformShapeInfo();
    zTadOffsets = packZ->platformOffsets();
  }

  // we get top K values first
  if (k == 1 && input->rankOf() > 1) {
    // k==1 optimization using IndexMax — only for multi-dimensional input.
    // For 1D input, applyIndexReduce with dims={0} produces a scalar result
    // which doesn't match the [1]-shaped indices output, causing memory errors.
    std::vector<LongType> dims = {input->rankOf() - 1};
    input->applyIndexReduce(indexreduce::IndexMax, indices, &dims);

    dim3 launchDims = getLaunchDims("top_k_mover");
    // copy values on specified indices
    topValuesMover<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        input->specialBuffer(), xTadShapeInfo, xTadOffsets, indices->specialBuffer(),
        iTadShapeInfo, iTadOffsets, values->specialBuffer(), zTadShapeInfo,
        zTadOffsets, tadLength, numTads, k);
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "topValuesMover failed");

  } else {
    int scanWidth = 1;
    dim3 topKIndices2 = topKIndices(scanWidth, sizeof(X), sizeof(Y));
    indicesAlongDimension<X, Y><<<topKIndices2.y, topKIndices2.x, topKIndices2.z, *context->getCudaStream()>>>(
        input->specialBuffer(), xTadShapeInfo, xTadOffsets, indices->specialBuffer(),
        iTadShapeInfo, iTadOffsets, values->specialBuffer(), zTadShapeInfo,
        zTadOffsets, tadLength, numTads, k, scanWidth, needSort);
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "indicesAlongDimension failed");

  }

  // Clean up device memory for 1D case (after kernel completes via stream sync in caller)
  if (deviceZeroOffset != nullptr) {
    // During CUDA graph capture, stream sync is illegal. Stream ordering guarantees correctness.
    if (!tl_graphExecutionActive && !tl_dspReplayActive) {
      cudaStreamSynchronize(*context->getCudaStream());
      sd::memory::CudaMemoryPool::getInstance().free(deviceZeroOffset, topkDevId, *context->getCudaStream());
    }
  }

  return Status::OK;
}

Status topKFunctor(LaunchContext* context, NDArray* input, NDArray* values, NDArray* indices,
                       const LongType k, bool needSort) {
  PointersManager manager(context, "top_k");

  NDArray::prepareSpecialUse({values, indices}, {input});

  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), topKFunctor_,
                        (context, input, values, indices, k, needSort), SD_COMMON_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({values, indices}, {input});

  manager.synchronize();

  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
