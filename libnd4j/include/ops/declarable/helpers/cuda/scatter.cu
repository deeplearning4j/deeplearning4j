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
#include <helpers/TAD.h>
#include <ops/declarable/helpers/scatter.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// x - indices, y - contains number of bad indices, z - input/output
template <typename X>
SD_KERNEL static void checkIndicesCuda(const void *vx, const sd::LongType *xShapeInfo, sd::LongType *y,
                                       const sd::LongType *zShapeInfo, const int axis) {
  const auto x = reinterpret_cast<const X *>(vx);

  __shared__ sd::LongType xRank, *coords, xLastDim;
  __shared__ sd::LongType xLen, numOfBadIndxPerBlock;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<sd::LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    xLen = shape::length(xShapeInfo);

    numOfBadIndxPerBlock = 0;
  }
  __syncthreads();

  auto xCoords = coords + threadIdx.x * xRank;

  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
    shape::index2coords(i, xShapeInfo, xCoords);

    const sd::LongType currentInd = x[shape::getOffset(xShapeInfo, xCoords)];

    if (currentInd >= shape::sizeAt(zShapeInfo, axis == -1 ? xCoords[xRank - 1] : axis)) {
      printf("checkIndices cuda: out of range element %lld at index %lld \n", currentInd, i);
      sd::math::atomics::sd_atomicAdd<sd::LongType>(&numOfBadIndxPerBlock, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && numOfBadIndxPerBlock != 0)
    sd::math::atomics::sd_atomicAdd<sd::LongType>(y, numOfBadIndxPerBlock);
}

///////////////////////////////////////////////////////////////////
template <typename X>
static void checkIndicesCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                     const cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo,
                                     sd::LongType *y, const sd::LongType *zShapeInfo, const int axis) {
  checkIndicesCuda<X><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, y, zShapeInfo, axis);
}

///////////////////////////////////////////////////////////////////
sd::LongType checkIndices(sd::LaunchContext *context, const NDArray &indices, const NDArray &output, const int axis) {
  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (indices.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(sd::LongType) * indices.rankOf() + 256;
  dim3 scatterDimsIndices = scatterDimsCheckIndices(indices.lengthOf(),indices.rankOf());
  const auto xType = indices.dataType();

  PointersManager manager(context, "scatterNDcheckIndices");

  // scalar, initial value = 0
  NDArray numOfBadIndx(sd::DataType::INT64, context, true);

  NDArray::prepareSpecialUse({&numOfBadIndx}, {&indices});
  BUILD_SINGLE_SELECTOR(xType, checkIndicesCudaLauncher,
                        (scatterDimsIndices.y,scatterDimsIndices.x, scatterDimsIndices.z, context->getCudaStream(), indices.specialBuffer(),
                            indices.specialShapeInfo(), reinterpret_cast<sd::LongType *>(numOfBadIndx.specialBuffer()),
                            output.specialShapeInfo(), axis),
                        SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&numOfBadIndx}, {&indices});

  manager.synchronize();

  return numOfBadIndx.t<sd::LongType>(0);
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output
template <typename X, typename Y>
SD_KERNEL static void scatterLockCuda(const int opCode, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ sd::LongType xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim, *coords;
  __shared__ sd::LongType xLen, zLen;
  __shared__ bool is1Dcase, xySameStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<sd::LongType *>(shmem);

    xLen = shape::length(xShapeInfo);
    zLen = shape::length(zShapeInfo);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    if (is1Dcase) xySameStride = shape::stride(xShapeInfo)[xNonUnitDim] = shape::stride(yShapeInfo)[yNonUnitDim];
  }
  __syncthreads();

  sd::LongType yOffset, zOffset;
  sd::LongType zFirstCoord, *yCoords, *zCoords;

  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {
    if (!is1Dcase) {
      yCoords = coords + threadIdx.x * (yRank + zRank);
      zCoords = yCoords + yRank;
      shape::index2coords(i, zShapeInfo, zCoords);
    }

    for (sd::LongType j = 0; j < xLen; ++j) {
      if (is1Dcase) {
        yOffset = j * shape::stride(yShapeInfo)[yNonUnitDim];
        zFirstCoord = x[xySameStride ? yOffset : j * shape::stride(xShapeInfo)[xNonUnitDim]];

        if (i != zFirstCoord) continue;

        zOffset = i * shape::stride(zShapeInfo)[zNonUnitDim];
      }

      else {
        shape::index2coords(j, xShapeInfo, yCoords);  // first xRank coordinates in yCoords are the same for y and x

        zFirstCoord = x[shape::getOffset(xShapeInfo, yCoords)];

        if (zCoords[0] != zFirstCoord) continue;

        for (sd::LongType k = 0; k < yRank - xRank; ++k) yCoords[xRank + k] = zCoords[k + 1];

        yOffset = shape::getOffset(yShapeInfo, yCoords);
        zOffset = shape::getOffset(zShapeInfo, zCoords);
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
SD_KERNEL static void scatterCuda(const int opCode, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                  const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);
  __shared__ sd::LongType xRank, yRank, zRank, xNonUnitDim, yNonUnitDim, zNonUnitDim, *coords;
  __shared__ sd::LongType yLen;
  __shared__ bool is1Dcase, xySameStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<sd::LongType *>(shmem);

    yLen = shape::length(yShapeInfo);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    if (is1Dcase) xySameStride = shape::stride(xShapeInfo)[xNonUnitDim] = shape::stride(yShapeInfo)[yNonUnitDim];
  }
  __syncthreads();

  sd::LongType xOffset, yOffset, zOffset;
  sd::LongType *yCoords, *zCoords;

  if (!is1Dcase) {
    yCoords = coords + threadIdx.x * (yRank + zRank);
    zCoords = yCoords + yRank;
  }

  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {
    if (is1Dcase) {
      yOffset = i * shape::stride(yShapeInfo)[yNonUnitDim];
      zOffset = x[xySameStride ? yOffset : i * shape::stride(xShapeInfo)[xNonUnitDim]] *
                shape::stride(zShapeInfo)[zNonUnitDim];
    } else {
      shape::index2coords(i, yShapeInfo, yCoords);

      yOffset = shape::getOffset(yShapeInfo, yCoords);
      xOffset =
          shape::getOffset(xShapeInfo, yCoords);  // first xRank coordinates in yCoords are the same for y and x -> for
      // (sd::LongType j = 0; j < xRank; ++j) xCoords[j] = yCoords[j];

      zCoords[0] = x[xOffset];

      for (sd::LongType j = 0; j < yRank - xRank; ++j) zCoords[j + 1] = yCoords[xRank + j];

      zOffset = shape::getOffset(zShapeInfo, zCoords);
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
                                const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo,
                                void *vz, const sd::LongType *zShapeInfo, const bool lock) {
  if (lock)
    scatterLockCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy,
                                                                                  yShapeInfo, vz, zShapeInfo);
  else
    scatterCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo,
                                                                              vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatter(sd::LaunchContext *context, pairwise::Ops op, const NDArray &indices, const NDArray &updates,
             NDArray &output, const bool lock) {
  const auto xType = indices.dataType();
  const auto yType = updates.dataType();

  dim3 launchDims = scatterDims(lock ? output.lengthOf() : updates.lengthOf(),updates.rankOf() + output.rankOf());
  PointersManager manager(context, "scatter");

  NDArray::prepareSpecialUse({&output}, {&updates, &indices});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterCudaLauncher,
                        (launchDims.y,launchDims.x, launchDims.z, context->getCudaStream(), op,
                            indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(),
                            updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output
template <typename X, typename Y>
SD_KERNEL static void scatterNDLockCuda(const int opCode, const void *vx, const sd::LongType *xShapeInfo,
                                        const void *vy, const sd::LongType *yShapeInfo, void *vz,
                                        const sd::LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ sd::LongType xRank, yRank, zRank, biggerXYRank, xLastDim, *coords, xNonUnitDim, yNonUnitDim, zNonUnitDim;
  __shared__ sd::LongType zLen, len;
  __shared__ bool is1Dcase;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<sd::LongType *>(shmem);

    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);
    xLastDim = shape::sizeAt(xShapeInfo, -1);

    biggerXYRank = xRank > yRank ? xRank : yRank;

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));

    len = is1Dcase ? shape::length(xShapeInfo) : shape::length(xShapeInfo) / xLastDim;
    zLen = shape::length(zShapeInfo);
  }
  __syncthreads();

  sd::LongType yOffset, zOffset, xOffset;
  sd::LongType *yCoords, *zCoords;

  if (!is1Dcase) {
    yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
    zCoords = yCoords + biggerXYRank;
  }

  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i += gridDim.x * blockDim.x) {
    if (!is1Dcase) shape::index2coords(i, zShapeInfo, zCoords);

    for (sd::LongType j = 0; j < len; j++) {

      if (is1Dcase) {
        if (x[j * shape::stride(xShapeInfo)[xNonUnitDim]] != i) continue;

        yOffset = j * shape::stride(yShapeInfo)[yNonUnitDim];
        zOffset = i * shape::stride(zShapeInfo)[zNonUnitDim];
      } else {
        shape::index2coords(j, xRank - 1, shape::shapeOf(const_cast<sd::LongType *>(xShapeInfo)),
                            yCoords);  // first xRank-1 coordinates in yCoords are the same for y and x

        // first iteration
        yCoords[xRank - 1] = 0;
        xOffset = shape::getOffset(xShapeInfo, yCoords);
        if (zCoords[0] != x[xOffset]) continue;

        // rest iterations
        bool matched = true;
        for (sd::LongType k = 1; k < xLastDim; k++) {
          yCoords[xRank - 1] = k;
          xOffset += shape::stride(xShapeInfo)[xRank - 1];
          if (zCoords[k] != x[xOffset]) {
            matched = false;
            break;
          }
        }

        if (!matched) continue;

        for (sd::LongType k = xLastDim; k < zRank; ++k) yCoords[yRank - zRank + k] = zCoords[k];

        yOffset = shape::getOffset(yShapeInfo, yCoords);
        zOffset = shape::getOffset(zShapeInfo, zCoords);
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
SD_KERNEL static void scatterNDCuda(const int opCode, const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                    const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);

  __shared__ sd::LongType xRank, yRank, zRank, biggerXYRank, xLastDim, *coords, xNonUnitDim, yNonUnitDim, zNonUnitDim;
  __shared__ sd::LongType yLen;
  __shared__ bool is1Dcase;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<sd::LongType *>(shmem);

    yLen = shape::length(yShapeInfo);
    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);
    xLastDim = shape::sizeAt(xShapeInfo, -1);

    biggerXYRank = xRank > yRank ? xRank : yRank;

    xNonUnitDim = yNonUnitDim = zNonUnitDim = 0;

    is1Dcase = (shape::isCommonVector(zShapeInfo, zNonUnitDim) || shape::isScalar(zShapeInfo)) &&
               (shape::isCommonVector(yShapeInfo, yNonUnitDim) || shape::isScalar(yShapeInfo)) &&
               (shape::isCommonVector(xShapeInfo, xNonUnitDim) || shape::isScalar(xShapeInfo));
  }
  __syncthreads();

  sd::LongType yOffset, zOffset;
  sd::LongType *yCoords, *zCoords;

  yCoords = coords + threadIdx.x * (biggerXYRank + zRank);
  zCoords = yCoords + biggerXYRank;
  for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < yLen; i += gridDim.x * blockDim.x) {

    shape::index2coords(i, yShapeInfo, yCoords);

    yOffset = shape::getOffset(yShapeInfo, yCoords);

    if (yRank >= xRank)
      zCoords[xLastDim] = yCoords[xRank - 1];  // saving y coordinate, since it might be changed in next instructions

    for (sd::LongType j = 0; j < xLastDim; ++j) {  // first xRank-1 coordinates in yCoords are the same for y and x
      yCoords[xRank - 1] = j;
      zCoords[j] = x[shape::getOffset(xShapeInfo, yCoords)];
    }

    for (sd::LongType j = xLastDim + 1; j < zRank; ++j) zCoords[j] = yCoords[yRank - zRank + j];

    zOffset = shape::getOffset(zShapeInfo, zCoords);

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
    } //end switch
  } //end for loop
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void scatterNDCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                  const cudaStream_t *stream, const int opCode, const void *vx,
                                  const sd::LongType *xShapeInfo, const void *vy, const sd::LongType *yShapeInfo,
                                  void *vz, const sd::LongType *zShapeInfo, const bool lock) {
  if (lock)
    scatterNDLockCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy,
                                                                                    yShapeInfo, vz, zShapeInfo);
  else
    scatterNDCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo,
                                                                                vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterND(sd::LaunchContext *context, pairwise::Ops op, const NDArray &indices, const NDArray &updates,
               NDArray &output, const bool lock) {
  const int xRank = indices.rankOf();
  const int yRank = updates.rankOf();
  const int zRank = output.rankOf();

  dim3 launchDims = scatterNdDims(lock ? output.lengthOf() : updates.lengthOf(),((yRank > xRank ? yRank : xRank) + zRank));
  const auto xType = indices.dataType();
  const auto yType = updates.dataType();

  PointersManager manager(context, "scatterND");

  NDArray::prepareSpecialUse({&output}, {&updates, &indices});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDCudaLauncher,
                        (launchDims.y,launchDims.x, launchDims.z, context->getCudaStream(), op,
                            indices.specialBuffer(), indices.specialShapeInfo(), updates.specialBuffer(),
                            updates.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), lock),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL void scatterForLossCuda(const void *vx, const sd::LongType *xShapeInfo, void *vy,
                                  const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<Z *>(vy);
  auto z = reinterpret_cast<Z *>(vz);

  __shared__ sd::LongType xLen;
  __shared__ sd::LongType xRank, *sharedMem;  // xRank = zRank, yRank = xRank + 1

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<sd::LongType *>(shmem);

    xLen = shape::length(xShapeInfo);
    xRank = shape::rank(xShapeInfo);
  }
  __syncthreads();

  const sd::LongType xInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (xInd >= xLen) return;

  sd::LongType *coords = sharedMem + threadIdx.x * (xRank + 1);

  shape::index2coords(xInd, xShapeInfo, coords);

  // y last coordinate
  coords[xRank] = x[shape::getOffset(xShapeInfo, coords)];

  const auto yOffset = shape::getOffset(yShapeInfo, coords);

  if (z == nullptr) {  // gradient calculation
    y[yOffset] -= 1.f;
  } else {
    z[shape::getOffset(zShapeInfo, coords)] = y[yOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void scatterForLossCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t *stream, const void *vx, const sd::LongType *xShapeInfo,
                                       void *vy, const sd::LongType *yShapeInfo, void *vz,
                                       const sd::LongType *zShapeInfo) {
  scatterForLossCuda<X, Z>
  <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
void scatterForLoss(sd::LaunchContext *context, const NDArray &indices, NDArray &updates, NDArray &output,
                    const bool calcGrad) {
  // shapes of indices and output must be the same
  // shape of indices should be the same as updates shape with last dimension excluded, for example if updates is
  // {a,b,c} then indices should be {a,b}

  PointersManager manager(context, "scatterForLoss");

  dim3 launchDIms = scatterDims(indices.lengthOf(),updates.rankOf());
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

