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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/scatter.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// x - indices, y - contains number of bad indices, z - input/output
template <typename X>
SD_KERNEL static void checkIndicesCuda(const void *vx, const LongType *xShapeInfo, LongType *y,
                                       const LongType *zShapeInfo, const int axis) {
  const auto x = reinterpret_cast<const X *>(vx);

  __shared__ LongType xRank, xLen, numOfBadIndxPerBlock;
  __shared__ const LongType *xShape, *xStride, *zShape;
  __shared__ LongType *coords;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    xLen = shape::length(xShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    numOfBadIndxPerBlock = 0;
  }
  __syncthreads();

  auto xCoords = coords + threadIdx.x * xRank;

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    INDEX2COORDS(i, xRank, xShape, xCoords);

    LongType xOffset;
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);

    const LongType currentInd = x[xOffset];

    const LongType limit = shape::sizeAt(zShapeInfo, axis == -1 ? xCoords[xRank - 1] : axis);
    if (currentInd >= limit) {
      sd::math::atomics::sd_atomicAdd<LongType>(&numOfBadIndxPerBlock, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && numOfBadIndxPerBlock != 0) {
    sd::math::atomics::sd_atomicAdd<LongType>(y, numOfBadIndxPerBlock);
  }
}

///////////////////////////////////////////////////////////////////
template <typename X>
static void checkIndicesCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo,
                                     LongType *y, const LongType *zShapeInfo, const int axis) {
  checkIndicesCuda<X><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, y, zShapeInfo, axis);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "checkIndicesCuda failed");
}

///////////////////////////////////////////////////////////////////
LongType checkIndices(LaunchContext *context, NDArray&indices, NDArray&output, const int axis) {
  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (indices.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(LongType) * indices.rankOf() + 256;
  dim3 scatterDimsIndices = scatterDimsCheckIndices(indices.lengthOf(), indices.rankOf());
  const auto xType = indices.dataType();

  PointersManager manager(context, "scatterNDcheckIndices");

  // scalar, initial value = 0
  NDArray numOfBadIndx(INT64, context, true);

  NDArray::prepareSpecialUse({&numOfBadIndx}, {&indices});
  BUILD_SINGLE_SELECTOR(
      xType, checkIndicesCudaLauncher,
      (scatterDimsIndices.y, scatterDimsIndices.x, scatterDimsIndices.z, context->getCudaStream(),
       indices.specialBuffer(), indices.specialShapeInfo(),
       reinterpret_cast<sd::LongType *>(numOfBadIndx.specialBuffer()), output.specialShapeInfo(), axis),
      SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&numOfBadIndx}, {&indices});

  manager.synchronize();

  return numOfBadIndx.t<LongType>(0);
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template <typename X, typename Y>
SD_KERNEL static void scatterLockCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                      const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ LongType xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim;
  __shared__ const LongType *xShape, *yShape, *zShape, *xStride, *yStride, *zStride;
  __shared__ LongType xLen, zLen;
  __shared__ bool is1Dcase, xySameStride;
  __shared__ LongType *coords;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    xStride = shape::stride(xShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zStride = shape::stride(zShapeInfo);

    xLen = shape::length(xShapeInfo);
    zLen = shape::length(zShapeInfo);

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    if (is1Dcase) xySameStride = xStride[xNonUnitDim] == yStride[yNonUnitDim];
  }
  __syncthreads();

  LongType yOffset, zOffset;
  LongType zFirstCoord, *yCoords, *zCoords;

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {
    if (!is1Dcase) {
      yCoords = coords + threadIdx.x * (yRank + zRank);
      zCoords = yCoords + yRank;
      INDEX2COORDS(i, zRank, zShape, zCoords);
    }

    for (LongType j = 0; j < xLen; ++j) {
      if (is1Dcase) {
        yOffset = j * yStride[yNonUnitDim];
        zFirstCoord = x[xySameStride ? yOffset : j];

        if (i != zFirstCoord) continue;

        zOffset = i * zStride[zNonUnitDim];
      } else {
        INDEX2COORDS(j, xRank, xShape, yCoords);

        LongType xOffset;
        COORDS2INDEX(xRank, xStride, yCoords, xOffset);
        zFirstCoord = x[xOffset];

        if (zCoords[0] != zFirstCoord) continue;

        for (LongType k = 0; k < yRank - xRank; ++k) yCoords[xRank + k] = zCoords[k + 1];

        COORDS2INDEX(yRank, yStride, yCoords, yOffset);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);
      }

      switch (opCode) {
        case pairwise::Add:
          z[zOffset] += y[yOffset];
          break;
        case pairwise::Subtract:
          z[zOffset] -= y[yOffset];
          break;
        case pairwise::Multiply:
          z[zOffset] *= y[yOffset];
          break;
        case pairwise::Divide:
          z[zOffset] /= y[yOffset];
          break;
        case pairwise::ReverseSubtract:
          z[zOffset] = y[yOffset] - z[zOffset];
          break;
        case pairwise::ReverseDivide:
          z[zOffset] = y[yOffset] / z[zOffset];
          break;
        case pairwise::CopyPws:
          z[zOffset] = y[yOffset];
          break;
        case pairwise::MaxPairwise:
          if (z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
          break;
        case pairwise::MinPairwise:
          if (z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
          break;
        default:
          continue;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template <typename X, typename Y>
SD_KERNEL static void scatterCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                  const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ LongType xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim;
  __shared__ const LongType *xShape, *yShape, *zShape, *xStride, *yStride, *zStride;
  __shared__ LongType yLen;
  __shared__ bool is1Dcase, xySameStride;
  __shared__ LongType *coords;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    xStride = shape::stride(xShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zStride = shape::stride(zShapeInfo);

    yLen = shape::length(yShapeInfo);

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    if (is1Dcase) xySameStride = xStride[xNonUnitDim] == yStride[yNonUnitDim];
  }
  __syncthreads();

  LongType xOffset, yOffset, zOffset;
  LongType *yCoords, *zCoords;

  if (!is1Dcase) {
    yCoords = coords + threadIdx.x * (yRank + zRank);
    zCoords = yCoords + yRank;
  }

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {
    if (is1Dcase) {
      yOffset = i * yStride[yNonUnitDim];
      zOffset = x[xySameStride ? yOffset : i * xStride[xNonUnitDim]] * zStride[zNonUnitDim];
    } else {
      INDEX2COORDS(i, yRank, yShape, yCoords);

      COORDS2INDEX(yRank, yStride, yCoords, yOffset);
      COORDS2INDEX(xRank, xStride, yCoords, xOffset);

      zCoords[0] = x[xOffset];

      for (LongType j = 0; j < yRank - xRank; ++j) {
        zCoords[j + 1] = yCoords[xRank + j];
      }

      COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    }

    switch (opCode) {
      case pairwise::Add:
        z[zOffset] += y[yOffset];
        break;
      case pairwise::Subtract:
        z[zOffset] -= y[yOffset];
        break;
      case pairwise::Multiply:
        z[zOffset] *= y[yOffset];
        break;
      case pairwise::Divide:
        z[zOffset] /= y[yOffset];
        break;
      case pairwise::ReverseSubtract:
        z[zOffset] = y[yOffset] - z[zOffset];
        break;
      case pairwise::ReverseDivide:
        z[zOffset] = y[yOffset] / z[zOffset];
        break;
      case pairwise::CopyPws:
        z[zOffset] = y[yOffset];
        break;
      case pairwise::MaxPairwise:
        if (z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
        break;
      case pairwise::MinPairwise:
        if (z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
        break;
      default:
        continue;
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void scatterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                const cudaStream_t *stream, const int opCode, const void *vx,
                                const LongType *xShapeInfo, const void *vy, const LongType *yShapeInfo, void *vz,
                                const LongType *zShapeInfo, const bool lock) {
  if (lock)
    scatterLockCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy,
                                                                                  yShapeInfo, vz, zShapeInfo);
  else
    scatterCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo,
                                                                              vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "scatterLockCuda failed");
}

///////////////////////////////////////////////////////////////////
void scatter(LaunchContext *context, pairwise::Ops op, NDArray&indices, NDArray&updates, NDArray &output,
             const bool lock) {
  const auto xType = indices.dataType();
  const auto yType = updates.dataType();

  dim3 launchDims = scatterDims(lock ? output.lengthOf() : updates.lengthOf(), updates.rankOf() + output.rankOf());
  PointersManager manager(context, "scatter");

  NDArray::prepareSpecialUse({&output}, {&updates, &indices});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), op,
                         indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(),
                         updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template <typename X, typename Y>
SD_KERNEL static void scatterNDLockCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                        const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ LongType xRank, yRank, zRank, biggerXYRank, xLastDim, xNonUnitDim, yNonUnitDim, zNonUnitDim;
  __shared__ const LongType *xShape, *yShape, *zShape, *xStride, *yStride, *zStride;
  __shared__ LongType zLen, len;
  __shared__ bool is1Dcase;
  __shared__ LongType *coords;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);
    xLastDim = shape::sizeAt(xShapeInfo, -1);

    xShape = shape::shapeOf(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);

    xStride = shape::stride(xShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zStride = shape::stride(zShapeInfo);

    biggerXYRank = xRank > yRank ? xRank : yRank;

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    len = is1Dcase ? shape::length(xShapeInfo) : shape::length(xShapeInfo) / xLastDim;
    zLen = shape::length(zShapeInfo);
  }
  __syncthreads();

  LongType yOffset, zOffset, xOffset;
  LongType *yCoords, *zCoords;

  if (!is1Dcase) {
    yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
    zCoords = yCoords + biggerXYRank;
  }

  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {
    if (!is1Dcase) INDEX2COORDS(i, zRank, zShape, zCoords);

    for (LongType j = 0; j < len; j++) {
      if (is1Dcase) {
        if (x[j * xStride[xNonUnitDim]] != i) continue;

        COORDS2INDEX(yRank, yStride, yCoords, yOffset);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);
      } else {
        INDEX2COORDS(j, xRank - 1, xShape, yCoords);

        yCoords[xRank - 1] = 0;
        COORDS2INDEX(xRank, xStride, yCoords, xOffset);
        if (zCoords[0] != x[xOffset]) continue;

        bool matched = true;
        for (LongType k = 1; k < xLastDim; k++) {
          yCoords[xRank - 1] = k;
          COORDS2INDEX(xRank, xStride, yCoords, xOffset);
          if (zCoords[k] != x[xOffset]) {
            matched = false;
            break;
          }
        }

        if (!matched) continue;

        for (LongType k = xLastDim; k < zRank; ++k) yCoords[yRank - zRank + k] = zCoords[k];

        COORDS2INDEX(yRank, yStride, yCoords, yOffset);
        COORDS2INDEX(zRank, zStride, zCoords, zOffset);
      }

      switch (opCode) {
        case pairwise::Add:
          z[zOffset] += y[yOffset];
          break;
        case pairwise::Subtract:
          z[zOffset] -= y[yOffset];
          break;
        case pairwise::Multiply:
          z[zOffset] *= y[yOffset];
          break;
        case pairwise::Divide:
          z[zOffset] /= y[yOffset];
          break;
        case pairwise::ReverseSubtract:
          z[zOffset] = y[yOffset] - z[zOffset];
          break;
        case pairwise::ReverseDivide:
          z[zOffset] = y[yOffset] / z[zOffset];
          break;
        case pairwise::CopyPws:
          z[zOffset] = y[yOffset];
          break;
        case pairwise::MaxPairwise:
          if (z[zOffset] < y[yOffset]) z[zOffset] = y[yOffset];
          break;
        case pairwise::MinPairwise:
          if (z[zOffset] > y[yOffset]) z[zOffset] = y[yOffset];
          break;
        default:
          continue;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template <typename X, typename Y>
SD_KERNEL static void scatterNDCuda(const int opCode, const void* vx, const LongType* xShapeInfo, const void* vy,
                                    const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo) {
  // Cast input and output pointers
  const auto x = reinterpret_cast<const X*>(vx);
  const auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<Y*>(vz);

  // Shared memory for shape information and flags
  __shared__ LongType xRank, yRank, zRank, biggerXYRank, xLastDim, xNonUnitDim, yNonUnitDim, zNonUnitDim, yLen;
  __shared__ bool is1Dcase;

  // Shared memory for coordinates
  __shared__ LongType* coords;

  if (threadIdx.x == 0) {
    // Dynamically allocated shared memory
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType*>(shmem);

    // Initialize shared values
    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);
    xLastDim = shape::sizeAt(xShapeInfo, -1);
    yLen = shape::length(yShapeInfo);

    biggerXYRank = max(xRank, yRank);

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    // Check if the operation involves 1D cases
    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));
  }
  __syncthreads();

  // Dynamically allocated memory for local coordinates
  LongType* yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
  LongType* zCoords = yCoords + biggerXYRank;

  // Process each element in y
  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {
    LongType yOffset, zOffset;

    // Convert linear index to multi-dimensional coordinates for y
    INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), yCoords);
    COORDS2INDEX(yRank, shape::stride(yShapeInfo), yCoords, yOffset);

    // Save the last coordinate of y if needed
    if (yRank >= xRank) {
      zCoords[xLastDim] = yCoords[xRank - 1];
    }

    // Map y coordinates to x and z coordinates
    for (LongType j = 0; j < xLastDim; ++j) {
      yCoords[xRank - 1] = j;
      COORDS2INDEX(xRank, shape::stride(xShapeInfo), yCoords, zCoords[j]);
    }

    // Adjust remaining coordinates for z
    for (LongType j = xLastDim + 1; j < zRank; ++j) {
      zCoords[j] = yCoords[yRank - zRank + j];
    }

    // Compute linear index for z
    COORDS2INDEX(zRank, shape::stride(zShapeInfo), zCoords, zOffset);

    // Perform the operation based on opCode
    switch (opCode) {
      case pairwise::Add:
        z[zOffset] += y[yOffset];
        break;
      case pairwise::Subtract:
        z[zOffset] -= y[yOffset];
        break;
      case pairwise::Multiply:
        z[zOffset] *= y[yOffset];
        break;
      case pairwise::Divide:
        z[zOffset] /= y[yOffset];
        break;
      case pairwise::ReverseSubtract:
        z[zOffset] = y[yOffset] - z[zOffset];
        break;
      case pairwise::ReverseDivide:
        z[zOffset] = y[yOffset] / z[zOffset];
        break;
      case pairwise::CopyPws:
        z[zOffset] = y[yOffset];
        break;
      case pairwise::MaxPairwise:
        z[zOffset] = max(z[zOffset], y[yOffset]);
        break;
      case pairwise::MinPairwise:
        z[zOffset] = min(z[zOffset], y[yOffset]);
        break;
      default:
        break;
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void scatterNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                  const cudaStream_t *stream, const int opCode, const void *vx,
                                  const LongType *xShapeInfo, const void *vy, const LongType *yShapeInfo, void *vz,
                                  const LongType *zShapeInfo, const bool lock) {
  if (lock)
    scatterNDLockCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy,
                                                                                    yShapeInfo, vz, zShapeInfo);
  else
    scatterNDCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo,
                                                                                vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "scatterNDCuda failed");
}

///////////////////////////////////////////////////////////////////
void scatterND(LaunchContext *context, pairwise::Ops op, NDArray&indices, NDArray&updates,
               NDArray &output, const bool lock) {
  const int xRank = indices.rankOf();
  const int yRank = updates.rankOf();
  const int zRank = output.rankOf();

  dim3 launchDims =
      scatterNdDims(lock ? output.lengthOf() : updates.lengthOf(), ((yRank > xRank ? yRank : xRank) + zRank));
  const auto xType = indices.dataType();
  const auto yType = updates.dataType();

  PointersManager manager(context, "scatterND");

  NDArray::prepareSpecialUse({&output}, {&updates, &indices});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, context->getCudaStream(), op,
                         indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(),
                         updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL void scatterForLossCuda(const void* vx, const LongType* xShapeInfo, void* vy, const LongType* yShapeInfo,
                                  void* vz, const LongType* zShapeInfo) {
  // Cast input and output pointers
  const auto x = reinterpret_cast<const X*>(vx);
  auto y = reinterpret_cast<Z*>(vy);
  auto z = reinterpret_cast<Z*>(vz);

  // Shared memory for shape information and coordinates
  __shared__ LongType xLen;
  __shared__ LongType xRank;
  __shared__ const LongType* xShape;
  __shared__ const LongType* xStride;
  __shared__ const LongType* yStride;
  __shared__ const LongType* zStride;

  if (threadIdx.x == 0) {
    // Initialize shared memory variables
    xLen = shape::length(xShapeInfo);
    xRank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zStride = zShapeInfo ? shape::stride(zShapeInfo) : nullptr;
  }
  __syncthreads();

  // Calculate global thread index
  const LongType xInd = threadIdx.x + blockIdx.x * blockDim.x;

  // Return if the thread index exceeds the length of x
  if (xInd >= xLen) return;

  // Dynamically allocated shared memory for coordinates
  extern __shared__ unsigned char shmem[];
  auto coords = reinterpret_cast<LongType*>(shmem) + threadIdx.x * (xRank + 1);

  // Convert linear index to coordinates for x
  INDEX2COORDS(xInd, xRank, xShape, coords);

  // Calculate offset for x
  LongType xOffset;
  COORDS2INDEX(xRank, xStride, coords, xOffset);

  // Update the last coordinate with the value from x
  coords[xRank] = x[xOffset];

  // Calculate offset for y
  LongType yOffset;
  COORDS2INDEX(xRank + 1, yStride, coords, yOffset);

  if (z == nullptr) {
    // Gradient calculation
    y[yOffset] -= 1.f;
  } else {
    // Calculate offset for z
    LongType zOffset;
    COORDS2INDEX(xRank + 1, zStride, coords, zOffset);

    // Update z with the value from y
    z[zOffset] = y[yOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void scatterForLossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, void *vy,
                                       const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo) {
  scatterForLossCuda<X, Z>
      <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "scatterUpdateCuda failed");
}

///////////////////////////////////////////////////////////////////
void scatterForLoss(LaunchContext *context, NDArray&indices, NDArray &updates, NDArray &output,
                    const bool calcGrad) {
  // shapes of indices and output must be the same
  // shape of indices should be the same as updates shape with last dimension excluded, for example if updates is
  // {a,b,c} then indices should be {a,b}

  PointersManager manager(context, "scatterForLoss");

  dim3 launchDIms = scatterDims(indices.lengthOf(), updates.rankOf());
  if (calcGrad) {
    NDArray::prepareSpecialUse({&updates}, {&indices});
    BUILD_DOUBLE_SELECTOR(
        indices.dataType(), updates.dataType(), scatterForLossCudaLauncher,
        (launchDIms.y, launchDIms.x, launchDIms.z, context->getCudaStream(), indices.specialBuffer(),
         indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), nullptr, nullptr),
        SD_INDEXING_TYPES, SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({&updates}, {&indices});
  } else {
    NDArray::prepareSpecialUse({&output}, {&indices, &updates});
    BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher,
                          (launchDIms.y, launchDIms.x, launchDIms.z, context->getCudaStream(), indices.specialBuffer(),
                           indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(),
                           output.specialBuffer(), output.specialShapeInfo()),
                          SD_INDEXING_TYPES, SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({&output}, {&indices, &updates});
  }

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
