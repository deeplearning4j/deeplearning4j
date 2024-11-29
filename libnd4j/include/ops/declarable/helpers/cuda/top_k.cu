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

  __shared__ LongType sharedMem[SD_CUDA_BLOCK_SIZE];
  __shared__ X elemToCompare;
  __shared__ const X* xTad;
  __shared__ LongType idx, xTadLen;

  if (threadIdx.x == 0) {
    xTadLen = shape::length(xTadShapeInfo);

    xTad = reinterpret_cast<const X*>(vx) + xTadOffsets[blockIdx.x];
    LongType yCoords[SD_MAX_RANK];
    LongType yOffset;
    INDEX2COORDS(blockIdx.x, shape::rank(yShapeInfo), yShapeInfo, yCoords);
    COORDS2INDEX(shape::rank(yShapeInfo), shape::shapeOf(yShapeInfo), yCoords, yOffset);
    idx = y[yOffset];
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(idx, shape::rank(xTadShapeInfo), xTadShapeInfo, xCoords);
    COORDS2INDEX(shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords, xOffset);
    elemToCompare = xTad[xOffset];
  }

  __syncthreads();

  sharedMem[threadIdx.x] = 0;
  for (LongType i = threadIdx.x; i < xTadLen; i += blockDim.x) {
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(i, shape::rank(xTadShapeInfo), xTadShapeInfo, xCoords);
    COORDS2INDEX(shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords, xOffset);
    if (elemToCompare < xTad[xOffset]) ++sharedMem[threadIdx.x];
  }

  __syncthreads();

  // aggregate sum
  for (LongType activeThreads = blockDim.x / 2; activeThreads > 0; activeThreads /= 2) {
    if (threadIdx.x < activeThreads) sharedMem[threadIdx.x] += sharedMem[threadIdx.x + activeThreads];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    LongType zCoords[SD_MAX_RANK];
    LongType zOffset;
    INDEX2COORDS(blockIdx.x, shape::rank(zShapeInfo), zShapeInfo, zCoords);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords, zOffset);
    z[zOffset] = *sharedMem < k;
  }
}
///////////////////////////////////////////////////////////////////
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
  BUILD_DOUBLE_SELECTOR(
      xType, yType, inTopKCudaLauncher,
      (topkDims2.y,topkDims2.x, topkDims2.z, context->getCudaStream(), predictions->specialBuffer(),
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
      INDEX2COORDS(e, shape::rank(iTadShapeInfo), iTadShapeInfo, iCoords);
      COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), iCoords, iOffset);
      auto idx = i[iOffset];

      INDEX2COORDS(e, shape::rank(zTadShapeInfo), zTadShapeInfo, zCoords);
      COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zCoords, zOffset);

      INDEX2COORDS(idx, shape::rank(xTadShapeInfo), xTadShapeInfo, xCoords);
      COORDS2INDEX(shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords, xOffset);

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

  __shared__ X localMaximum;
  if (threadIdx.x == 0) localMaximum = -DataTypeUtils::max<X>();
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
        INDEX2COORDS(e, shape::rank(xTadShapeInfo), xTadShapeInfo, xCoords);
        COORDS2INDEX(shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), xCoords, xOffset);
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

      // at this point we have local part ready for merge and define global maximum for this iteration, and local
      // maximum for next iteration
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
        INDEX2COORDS(p, shape::rank(zTadShapeInfo), zTadShapeInfo, zCoords);
        COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zCoords, zOffset);
        z[zOffset] = tempValues[scanWidth - 1];
        LongType iCoords[SD_MAX_RANK];
        LongType iOffset;
        INDEX2COORDS(p, shape::rank(iTadShapeInfo), iTadShapeInfo, iCoords);
        COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), iCoords, iOffset);
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
              LongType t0Coords[SD_MAX_RANK];
              LongType t0Offset;
              INDEX2COORDS(top - 1, shape::rank(iTadShapeInfo), iTadShapeInfo, t0Coords);
              COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), t0Coords, t0Offset);
              LongType t1Coords[SD_MAX_RANK];
              LongType t1Offset;
              INDEX2COORDS(top, shape::rank(iTadShapeInfo), iTadShapeInfo, t1Coords);
              COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), t1Coords, t1Offset);

              if (i[t0Offset] > i[t1Offset]) {
                // swap indices first
                Y di0 = i[t0Offset];
                i[t0Offset] = i[t1Offset];
                i[t1Offset] = di0;

                // swap values next
                LongType zT0Coords[SD_MAX_RANK];
                LongType zT0Offset;
                INDEX2COORDS(top - 1, shape::rank(zTadShapeInfo), zTadShapeInfo, zT0Coords);
                COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zT0Coords, zT0Offset);
                LongType zT1Coords[SD_MAX_RANK];
                LongType zT1Offset;
                INDEX2COORDS(top, shape::rank(zTadShapeInfo), zTadShapeInfo, zT1Coords);
                COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zT1Coords, zT1Offset);

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
              LongType t0Coords[SD_MAX_RANK];
              LongType t0Offset;
              INDEX2COORDS(top - 1, shape::rank(iTadShapeInfo), iTadShapeInfo, t0Coords);
              COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), t0Coords, t0Offset);
              LongType t1Coords[SD_MAX_RANK];
              LongType t1Offset;
              INDEX2COORDS(top, shape::rank(iTadShapeInfo), iTadShapeInfo, t1Coords);
              COORDS2INDEX(shape::rank(iTadShapeInfo), shape::shapeOf(iTadShapeInfo), t1Coords, t1Offset);

              if (i[t0Offset] > i[t1Offset]) {
                // swap indices first
                Y di0 = i[t0Offset];
                i[t0Offset] = i[t1Offset];
                i[t1Offset] = di0;

                // swap values next
                LongType zT0Coords[SD_MAX_RANK];
                LongType zT0Offset;
                INDEX2COORDS(top - 1, shape::rank(zTadShapeInfo), zTadShapeInfo, zT0Coords);
                COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zT0Coords, zT0Offset);
                LongType zT1Coords[SD_MAX_RANK];
                LongType zT1Offset;
                INDEX2COORDS(top, shape::rank(zTadShapeInfo), zTadShapeInfo, zT1Coords);
                COORDS2INDEX(shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), zT1Coords, zT1Offset);

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
    input->applyIndexReduce(indexreduce::IndexMax, *indices, &dims);

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
