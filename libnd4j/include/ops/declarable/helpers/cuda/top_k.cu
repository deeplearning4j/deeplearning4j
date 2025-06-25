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
#include <ops/declarable/helpers/top_k.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"
#include <system/selective_rendering.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__global__ static void inTopKCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                  const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                  const LongType* xTadShapeInfo, const LongType* xTadOffsets,
                                  const LongType k) {
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<bool*>(vz);

  // Shared memory for caching shape information
  __shared__ LongType shared_xRank;
  __shared__ const LongType* shared_xShape;
  __shared__ const LongType* shared_xStride;

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
  __shared__ LongType idx;

  // Initialize shared memory
  if (threadIdx.x == 0) {
    // Cache ranks
    shared_xRank = shape::rank(xShapeInfo);
    shared_yRank = shape::rank(yShapeInfo);
    shared_zRank = shape::rank(zShapeInfo);
    shared_xTadRank = shape::rank(xTadShapeInfo);

    // Cache shapes
    shared_xShape = shape::shapeOf(xShapeInfo);
    shared_yShape = shape::shapeOf(yShapeInfo);
    shared_zShape = shape::shapeOf(zShapeInfo);
    shared_xTadShape = shape::shapeOf(xTadShapeInfo);

    // Cache strides
    shared_xStride = shape::stride(xShapeInfo);
    shared_yStride = shape::stride(yShapeInfo);
    shared_zStride = shape::stride(zShapeInfo);
    shared_xTadStride = shape::stride(xTadShapeInfo);

    // Cache xTad length
    xTadLen = shape::length(xTadShapeInfo);

    // Initialize xTad pointer
    // Assuming xTadOffsets is used to compute the starting point for each block
    // Adjusted to point to the correct location in the xTad
    // If xTadOffsets[blockIdx.x] is already in terms of elements, this is correct
    // Otherwise, multiply by the size of X if xTadOffsets are byte offsets
    // Here, we assume they are element offsets
    // If not, use: xTad = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x] / sizeof(X);
    // Adjust accordingly based on how xTadOffsets are defined
    const X* xTadPtr = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];

    // Compute y coordinates from blockIdx.x
    LongType yCoords[SD_MAX_RANK];
    LongType yOffset;
    INDEX2COORDS(blockIdx.x, shared_yRank, shared_yShape, yCoords);
    COORDS2INDEX(shared_yRank, shared_yStride, yCoords, yOffset);

    // Retrieve the index from y at the computed offset
    idx = y[yOffset];

    // Compute coordinates and offset for xTad using idx
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(idx, shared_xTadRank, shared_xTadShape, xCoords);
    COORDS2INDEX(shared_xTadRank, shared_xTadStride, xCoords, xOffset);

    // Store the element to compare
    elemToCompare = xTadPtr[xOffset];
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Initialize shared memory for reduction
  extern __shared__ LongType sharedMem[];
  sharedMem[threadIdx.x] = 0;
  __syncthreads();

  // Pointer to xTad data
  const X* xTad = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];

  // Iterate over xTad elements using cached shape info
  for (LongType i = threadIdx.x; i < xTadLen; i += blockDim.x) {
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;

    // Use cached rank, shape, and stride
    INDEX2COORDS(i, shared_xTadRank, shared_xTadShape, xCoords);
    COORDS2INDEX(shared_xTadRank, shared_xTadStride, xCoords, xOffset);

    // Compare and update shared memory
    if (elemToCompare < xTad[xOffset]) {
      sharedMem[threadIdx.x]++;
    }
  }

  // Ensure all threads have completed the counting
  __syncthreads();

  // Perform parallel reduction to sum counts
  for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
    if (threadIdx.x < activeThreads) {
      sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
    }
    __syncthreads();
  }

  // Write the result to z using cached shape info
  if (threadIdx.x == 0) {
    LongType zCoords[SD_MAX_RANK];
    LongType zOffset;

    // Compute z coordinates from blockIdx.x
    INDEX2COORDS(blockIdx.x, shared_zRank, shared_zShape, zCoords);
    COORDS2INDEX(shared_zRank, shared_zStride, zCoords, zOffset);

    // Compare the aggregated count with k and store the result
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

  const auto packX = ConstantTadHelper::getInstance().tadForDimensions(predictions->shapeInfo(), {1});

  dim3 topkDims2 = topkDims(packX->numberOfTads());
  const auto xType = predictions->dataType();
  const auto yType = targets->dataType();

  NDArray::prepareSpecialUse({output}, {predictions, targets});
#if SD_IS_PAIR_TYPE_COMPILED(xType,yType)
  BUILD_DOUBLE_SELECTOR(
      xType, yType, inTopKCudaLauncher,
      (topkDims2.y,topkDims2.x, topkDims2.z, context->getCudaStream(), predictions->specialBuffer(),
          predictions->specialShapeInfo(), targets->specialBuffer(), targets->specialShapeInfo(), output->specialBuffer(),
          output->specialShapeInfo(), packX->specialShapeInfo(), packX->specialOffsets(), k),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {predictions, targets});
#endif

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
      auto idx = i[iOffset];

      INDEX2COORDS(e, zRank, zShape, zCoords);
      COORDS2INDEX(zRank, zStride, zCoords, zOffset);

      INDEX2COORDS(idx, xRank, xShape, xCoords);
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

    // we'll do multiple reads here
    for (int p = 0; p < k; p += scanWidth) {
      // resetting temporary storage
      for (int p = 0; p < scanWidth; p++) {
        tempValues[p] = -DataTypeUtils::max<X>();
        tempIndices[p] = DataTypeUtils::max<Y>();
      }

      // local max values/indices
      for (int e = threadIdx.x; e < tadLength; e++) {
        LongType xCoords[SD_MAX_RANK];
        LongType xOffset;
        INDEX2COORDS(e, xRank, xShape, xCoords);
        COORDS2INDEX(xRank, xStride, xCoords, xOffset);
        auto value = x[xOffset];

        // we'll compare this value to current stored ones
        for (int f = 0; f < scanWidth; f++) {
          if (value > tempValues[f] && (p == 0 || value < localMaximum)) {
            tempValues[f] = value;
            tempIndices[f] = e;
          }
        }
      }
      __syncthreads();

      // at this point we have local part ready for merge and define global maximum for this iteration
      for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
        if (threadIdx.x < activeThreads) {
          if (tempValues[0] < tempValues[0 + activeThreads * scanWidth]) {
            tempValues[0] = tempValues[0 + activeThreads * scanWidth];
            tempIndices[0] = tempIndices[0 + activeThreads * scanWidth];
          }
        }
        __syncthreads();
      }
      __syncthreads();

      // at this point we know local minimum for next iteration
      if (threadIdx.x == 0) {
        localMaximum = tempValues[scanWidth - 1];
        LongType zCoords[SD_MAX_RANK];
        LongType zOffset;
        INDEX2COORDS(p, zRank, zShape, zCoords);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);
        z[zOffset] = tempValues[scanWidth - 1];
        LongType iCoords[SD_MAX_RANK];
        LongType iOffset;
        INDEX2COORDS(p, iRank, iShape, iCoords);
        COORDS2INDEX(iRank, iStride, iCoords, iOffset);
        i[iOffset] = tempIndices[scanWidth - 1];
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
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {input->rankOf() - 1});
  auto packI = ConstantTadHelper::getInstance().tadForDimensions(indices->shapeInfo(), {input->rankOf() - 1});
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(values->shapeInfo(), {input->rankOf() - 1});

  auto tadLength = shape::length(packX->primaryShapeInfo());

  // we get top K values first
  if (k == 1) {
    std::vector<LongType> dims = {input->rankOf() - 1};
    input->applyIndexReduce(indexreduce::IndexMax, indices, &dims);

    dim3 launchDims = getLaunchDims("top_k_mover");
    // copy values on specified indices
    topValuesMover<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
        input->specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), indices->specialBuffer(),
        packI->platformShapeInfo(), packI->platformOffsets(), values->specialBuffer(), packZ->platformShapeInfo(),
        packZ->platformOffsets(), tadLength, packX->numberOfTads(), k);
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "topValuesMover failed");

  } else {
    int scanWidth = 1;
    dim3 topKIndices2 = topKIndices(scanWidth, sizeof(X), sizeof(Y));
    indicesAlongDimension<X, Y><<<topKIndices2.y, topKIndices2.x, topKIndices2.z, *context->getCudaStream()>>>(
        input->specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), indices->specialBuffer(),
        packI->platformShapeInfo(), packI->platformOffsets(), values->specialBuffer(), packZ->platformShapeInfo(),
        packZ->platformOffsets(), tadLength, packX->numberOfTads(), k, scanWidth, needSort);
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "indicesAlongDimension failed");

  }

  return Status::OK;
}

Status topKFunctor(LaunchContext* context, NDArray* input, NDArray* values, NDArray* indices,
                       const LongType k, bool needSort) {
  input->syncToDevice();

  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), topKFunctor_,
                        (context, input, values, indices, k, needSort), SD_COMMON_TYPES, SD_INDEXING_TYPES);

  values->tickWriteDevice();
  indices->tickWriteDevice();

  return Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
