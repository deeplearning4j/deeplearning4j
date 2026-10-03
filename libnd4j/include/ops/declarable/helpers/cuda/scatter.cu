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
    if (currentInd < 0 || currentInd >= limit) {
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
      (scatterDimsIndices.x, scatterDimsIndices.y, scatterDimsIndices.z, context->getCudaStream(),
       indices.specialBuffer(), indices.specialShapeInfo(),
       reinterpret_cast<sd::LongType *>(numOfBadIndx.specialBuffer()), output.specialShapeInfo(), axis),
      SD_INTEGER_TYPES);
  NDArray::registerSpecialUse({&numOfBadIndx}, {&indices});

  manager.synchronize();

  return numOfBadIndx.t<LongType>(0);
}

///////////////////////////////////////////////////////////////////
// Applies update y to output element *z. An ordered kernel gives each output element one thread, so it
// updates in place; a per-update kernel meets a repeated destination only for Add and Subtract, which it
// accumulates atomically.
template <typename Y>
static SD_DEVICE void applyScatterUpdate(const int opCode, Y *z, const Y y, const bool atomicAccumulate) {
  switch (opCode) {
    case pairwise::Add:
      if (atomicAccumulate)
        sd::math::atomics::sd_atomicAdd<Y>(z, y);
      else
        *z += y;
      break;
    case pairwise::Subtract:
      if (atomicAccumulate)
        sd::math::atomics::sd_atomicAdd<Y>(z, static_cast<Y>(-y));
      else
        *z -= y;
      break;
    case pairwise::Multiply:
      *z *= y;
      break;
    case pairwise::Divide:
      *z /= y;
      break;
    case pairwise::ReverseSubtract:
      *z = y - *z;
      break;
    case pairwise::ReverseDivide:
      *z = y / *z;
      break;
    case pairwise::CopyPws:
      *z = y;
      break;
    case pairwise::MaxPairwise:
      *z = sd::math::sd_max<Y>(*z, y);
      break;
    case pairwise::MinPairwise:
      *z = sd::math::sd_min<Y>(*z, y);
      break;
    default:
      break;
  }
}

///////////////////////////////////////////////////////////////////
// scatter: index k, x's k-th element in logical order, names slice x[k] of z along its first dimension
// (numSlices of them, sliceLen elements each); update slice k is y's k-th run of sliceLen elements in logical
// order, and element p of a slice is its p-th in logical order. That pairing serves every updates layout the
// ops accept: x.shape + z.shape[1:], [x.length] + z.shape[1:] for vector indices, and x's shape for a vector
// z. Indices outside [0, numSlices) are skipped.

// x - indices, y - updates, z - input/output. One thread per update. With repeated given, runs only when
// *repeated is zero (scatterLockCuda runs otherwise).
template <typename X, typename Y>
SD_KERNEL static void scatterCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                  const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                  const LongType numSlices, const LongType sliceLen, const int *repeated) {
  if (repeated != nullptr && *repeated != 0) return;
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);
  const int xRank = shape::rank(xShapeInfo);
  const int yRank = shape::rank(yShapeInfo);
  const int zRank = shape::rank(zShapeInfo);
  const LongType yLen = shape::length(yShapeInfo);
  LongType coords[SD_MAX_RANK];

  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < yLen;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType xOffset, yOffset, zOffset;
    INDEX2COORDS(i / sliceLen, xRank, shape::shapeOf(xShapeInfo), coords);
    COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
    const auto slice = static_cast<LongType>(x[xOffset]);
    if (slice < 0 || slice >= numSlices) continue;

    INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), coords);
    COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
    INDEX2COORDS(slice * sliceLen + i % sliceLen, zRank, shape::shapeOf(zShapeInfo), coords);
    COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
    applyScatterUpdate<Y>(opCode, &z[zOffset], y[yOffset], true);
  }
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - input/output. Thread p owns element p of every slice and walks the indices in
// order, so updates sharing a destination apply one after another in index order, as on the CPU. With
// repeated given, runs only when *repeated is non-zero (scatterCuda runs otherwise).
template <typename X, typename Y>
SD_KERNEL static void scatterLockCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                      const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                      const LongType numSlices, const LongType sliceLen, const int *repeated) {
  if (repeated != nullptr && *repeated == 0) return;
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);
  const int xRank = shape::rank(xShapeInfo);
  const int yRank = shape::rank(yShapeInfo);
  const int zRank = shape::rank(zShapeInfo);
  const LongType xLen = shape::length(xShapeInfo);
  LongType coords[SD_MAX_RANK];

  for (LongType p = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; p < sliceLen;
       p += static_cast<LongType>(gridDim.x) * blockDim.x) {
    for (LongType k = 0; k < xLen; ++k) {
      LongType xOffset, yOffset, zOffset;
      INDEX2COORDS(k, xRank, shape::shapeOf(xShapeInfo), coords);
      COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
      const auto slice = static_cast<LongType>(x[xOffset]);
      if (slice < 0 || slice >= numSlices) continue;

      INDEX2COORDS(k * sliceLen + p, yRank, shape::shapeOf(yShapeInfo), coords);
      COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
      INDEX2COORDS(slice * sliceLen + p, zRank, shape::shapeOf(zShapeInfo), coords);
      COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
      applyScatterUpdate<Y>(opCode, &z[zOffset], y[yOffset], false);
    }
  }
}

///////////////////////////////////////////////////////////////////
// x - indices, each naming a destination in [0, numDestinations). Sets each destination's bit in marks and
// *repeated when a bit was already set.
template <typename X>
SD_KERNEL static void markRepeatedDestinationsCuda(const void *vx, const LongType *xShapeInfo,
                                                   const LongType numDestinations, unsigned int *marks,
                                                   int *repeated) {
  const auto x = reinterpret_cast<const X *>(vx);
  const LongType xLen = shape::length(xShapeInfo);
  const int xRank = shape::rank(xShapeInfo);
  LongType coords[SD_MAX_RANK];

  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < xLen;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType xOffset;
    INDEX2COORDS(i, xRank, shape::shapeOf(xShapeInfo), coords);
    COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
    const auto destination = static_cast<LongType>(x[xOffset]);
    if (destination < 0 || destination >= numDestinations) continue;
    const unsigned int bit = 1u << (destination & 31);
    if ((atomicOr(&marks[destination >> 5], bit) & bit) != 0) *repeated = 1;
  }
}

template <typename X>
static void markRepeatedDestinationsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                                 const cudaStream_t *stream, const void *vx,
                                                 const LongType *xShapeInfo, const LongType numDestinations,
                                                 unsigned int *marks, int *repeated) {
  markRepeatedDestinationsCuda<X><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, xShapeInfo, numDestinations,
                                                                                  marks, repeated);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "markRepeatedDestinationsCuda failed");
}

///////////////////////////////////////////////////////////////////
// lock: scatterLockCuda applies every update in index order. Otherwise scatterCuda applies one update per
// thread, and when repeated is given both kernels run, its device value choosing the one that does the work.
template <typename X, typename Y>
static void scatterCudaLauncher(const dim3 &perUpdateDims, const dim3 &orderedDims, const cudaStream_t *stream,
                                const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                const LongType numSlices, const LongType sliceLen, const bool lock,
                                const int *repeated) {
  if (!lock)
    scatterCuda<X, Y><<<perUpdateDims.x, perUpdateDims.y, 0, *stream>>>(opCode, vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                       zShapeInfo, numSlices, sliceLen, repeated);
  if (lock || repeated != nullptr)
    scatterLockCuda<X, Y><<<orderedDims.x, orderedDims.y, 0, *stream>>>(
        opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, numSlices, sliceLen, lock ? nullptr : repeated);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "scatterCuda failed");
}

///////////////////////////////////////////////////////////////////
void scatter(LaunchContext *context, pairwise::Ops op, NDArray&indices, NDArray&updatesIn, NDArray &output,
             const bool lock) {
  if (indices.lengthOf() == 0 || updatesIn.lengthOf() == 0 || output.lengthOf() == 0) return;
  // The kernels read the updates in output's type.
  NDArray *castUpdates = updatesIn.dataType() == output.dataType() ? nullptr : updatesIn.cast(output.dataType());
  NDArray &updates = castUpdates != nullptr ? *castUpdates : updatesIn;
  const auto xType = indices.dataType();
  const auto yType = output.dataType();
  const LongType numSlices = output.rankOf() == 0 ? 1 : output.sizeAt(0);
  const LongType sliceLen = output.lengthOf() / numSlices;
  PointersManager manager(context, "scatter");

  // Repeated indices give several updates one destination. scatterCuda accumulates addition and
  // subtraction atomically. For any other op markRepeatedDestinationsCuda flags a repeated destination
  // on the device, and the flag hands the updates to scatterLockCuda, which applies them one at a time
  // in index order, as the CPU helper does. The flag is never read on the host: no synchronization,
  // and the choice is recorded under graph capture.
  NDArray *marks = nullptr, *repeated = nullptr;
  if (!lock && op != pairwise::Add && op != pairwise::Subtract && indices.lengthOf() > 1) {
    std::vector<LongType> marksShape = {(numSlices + 31) / 32};
    marks = new NDArray('c', marksShape, INT32, context);
    repeated = new NDArray(INT32, context, true);
    marks->nullify();
    repeated->nullify();
    dim3 markDims = scatterDims(indices.lengthOf(), indices.rankOf());

    NDArray::prepareSpecialUse({marks, repeated}, {&indices});
    BUILD_SINGLE_SELECTOR(xType, markRepeatedDestinationsCudaLauncher,
                          (markDims.x, markDims.y, context->getCudaStream(), indices.specialBuffer(),
                           indices.specialShapeInfo(), numSlices,
                           reinterpret_cast<unsigned int *>(marks->specialBuffer()),
                           reinterpret_cast<int *>(repeated->specialBuffer())),
                          SD_INDEXING_TYPES);
    NDArray::registerSpecialUse({marks, repeated}, {&indices});
  }

  dim3 perUpdateDims = scatterDims(updates.lengthOf(), updates.rankOf());
  dim3 orderedDims = scatterDims(sliceLen, updates.rankOf());
  const int *repeatedFlag = repeated != nullptr ? reinterpret_cast<const int *>(repeated->specialBuffer()) : nullptr;

  NDArray::prepareSpecialUse({&output}, {&updates, &indices, repeated});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterCudaLauncher,
                        (perUpdateDims, orderedDims, context->getCudaStream(), op, indices.specialBuffer(),
                         indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(),
                         output.specialBuffer(), output.specialShapeInfo(), numSlices, sliceLen, lock, repeatedFlag),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices, repeated});

  manager.synchronize();
  delete marks;
  delete repeated;
  delete castUpdates;
}

///////////////////////////////////////////////////////////////////
// scatterND: index row r, x's r-th run of indexLength elements in logical order, names z's leading
// indexLength coordinates, flattened into destination slice d (sliceLen elements each); update slice r is
// y's r-th run of sliceLen elements, and element p of a slice is its p-th in logical order. Rows with a
// coordinate out of range are skipped.

// The destination slice of index row r, or -1 when a coordinate is out of range.
template <typename X>
static SD_DEVICE LongType scatterNdDestination(const X *x, const LongType *xShapeInfo, const LongType *zShapeInfo,
                                               const LongType indexLength, const LongType row, LongType *coords) {
  const int xRank = shape::rank(xShapeInfo);
  const LongType *zShape = shape::shapeOf(zShapeInfo);
  LongType destination = 0;
  for (LongType j = 0; j < indexLength; ++j) {
    LongType xOffset;
    INDEX2COORDS(row * indexLength + j, xRank, shape::shapeOf(xShapeInfo), coords);
    COORDS2INDEX(xRank, shape::stride(xShapeInfo), coords, xOffset);
    const auto index = static_cast<LongType>(x[xOffset]);
    if (index < 0 || index >= zShape[j]) return -1;
    destination = destination * zShape[j] + index;
  }
  return destination;
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output. One thread per update. With repeated given, runs only when *repeated
// is zero (scatterNDLockCuda runs otherwise).
template <typename X, typename Y>
SD_KERNEL static void scatterNDCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                    const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                    const LongType indexLength, const LongType sliceLen, const int *repeated) {
  if (repeated != nullptr && *repeated != 0) return;
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);
  const int yRank = shape::rank(yShapeInfo);
  const int zRank = shape::rank(zShapeInfo);
  const LongType yLen = shape::length(yShapeInfo);
  LongType coords[SD_MAX_RANK];

  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < yLen;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType destination = scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, i / sliceLen, coords);
    if (destination < 0) continue;

    LongType yOffset, zOffset;
    INDEX2COORDS(i, yRank, shape::shapeOf(yShapeInfo), coords);
    COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
    INDEX2COORDS(destination * sliceLen + i % sliceLen, zRank, shape::shapeOf(zShapeInfo), coords);
    COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
    applyScatterUpdate<Y>(opCode, &z[zOffset], y[yOffset], true);
  }
}

///////////////////////////////////////////////////////////////////
// x - indices, y - updates, z - output. Thread p owns element p of every destination slice and walks the
// index rows in order, as scatterLockCuda does. With repeated given, runs only when *repeated is non-zero
// (scatterNDCuda runs otherwise).
template <typename X, typename Y>
SD_KERNEL static void scatterNDLockCuda(const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                        const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                        const LongType indexLength, const LongType sliceLen, const int *repeated) {
  if (repeated != nullptr && *repeated == 0) return;
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Y *>(vz);
  const int yRank = shape::rank(yShapeInfo);
  const int zRank = shape::rank(zShapeInfo);
  const LongType rows = shape::length(xShapeInfo) / indexLength;
  LongType coords[SD_MAX_RANK];

  for (LongType p = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; p < sliceLen;
       p += static_cast<LongType>(gridDim.x) * blockDim.x) {
    for (LongType r = 0; r < rows; ++r) {
      const LongType destination = scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, r, coords);
      if (destination < 0) continue;

      LongType yOffset, zOffset;
      INDEX2COORDS(r * sliceLen + p, yRank, shape::shapeOf(yShapeInfo), coords);
      COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);
      INDEX2COORDS(destination * sliceLen + p, zRank, shape::shapeOf(zShapeInfo), coords);
      COORDS2INDEX(zRank, shape::stride(zShapeInfo), coords, zOffset);
      applyScatterUpdate<Y>(opCode, &z[zOffset], y[yOffset], false);
    }
  }
}

///////////////////////////////////////////////////////////////////
// x - indices. Sets the bit of each index row's destination slice in marks and *repeated when a bit was
// already set.
template <typename X>
SD_KERNEL static void markRepeatedNdDestinationsCuda(const void *vx, const LongType *xShapeInfo,
                                                     const LongType *zShapeInfo, const LongType indexLength,
                                                     unsigned int *marks, int *repeated) {
  const auto x = reinterpret_cast<const X *>(vx);
  const LongType rows = shape::length(xShapeInfo) / indexLength;
  LongType coords[SD_MAX_RANK];

  for (LongType row = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType destination = scatterNdDestination<X>(x, xShapeInfo, zShapeInfo, indexLength, row, coords);
    if (destination < 0) continue;
    const unsigned int bit = 1u << (destination & 31);
    if ((atomicOr(&marks[destination >> 5], bit) & bit) != 0) *repeated = 1;
  }
}

template <typename X>
static void markRepeatedNdDestinationsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                                   const cudaStream_t *stream, const void *vx,
                                                   const LongType *xShapeInfo, const LongType *zShapeInfo,
                                                   const LongType indexLength, unsigned int *marks,
                                                   int *repeated) {
  markRepeatedNdDestinationsCuda<X><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(vx, xShapeInfo, zShapeInfo,
                                                                                    indexLength, marks, repeated);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "markRepeatedNdDestinationsCuda failed");
}

///////////////////////////////////////////////////////////////////
// As scatterCudaLauncher, with scatterNDCuda and scatterNDLockCuda.
template <typename X, typename Y>
static void scatterNDCudaLauncher(const dim3 &perUpdateDims, const dim3 &orderedDims, const cudaStream_t *stream,
                                  const int opCode, const void *vx, const LongType *xShapeInfo, const void *vy,
                                  const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo,
                                  const LongType indexLength, const LongType sliceLen, const bool lock,
                                  const int *repeated) {
  if (!lock)
    scatterNDCuda<X, Y><<<perUpdateDims.x, perUpdateDims.y, 0, *stream>>>(
        opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, indexLength, sliceLen, repeated);
  if (lock || repeated != nullptr)
    scatterNDLockCuda<X, Y><<<orderedDims.x, orderedDims.y, 0, *stream>>>(
        opCode, vx, xShapeInfo, vy, yShapeInfo, vz, zShapeInfo, indexLength, sliceLen, lock ? nullptr : repeated);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "scatterNDCuda failed");
}

///////////////////////////////////////////////////////////////////
void scatterND(LaunchContext *context, pairwise::Ops op, NDArray&indices, NDArray&updatesIn,
               NDArray &output, const bool lock) {
  if (indices.lengthOf() == 0 || updatesIn.lengthOf() == 0 || output.lengthOf() == 0) return;
  // The kernels read the updates in output's type.
  NDArray *castUpdates = updatesIn.dataType() == output.dataType() ? nullptr : updatesIn.cast(output.dataType());
  NDArray &updates = castUpdates != nullptr ? *castUpdates : updatesIn;
  const auto xType = indices.dataType();
  const auto yType = output.dataType();
  const LongType indexLength = indices.sizeAt(-1);
  const LongType rows = indices.lengthOf() / indexLength;
  LongType numDestinations = 1;
  for (LongType j = 0; j < indexLength; ++j) numDestinations *= output.sizeAt(j);
  const LongType sliceLen = output.lengthOf() / numDestinations;
  PointersManager manager(context, "scatterND");

  // As in scatter(): addition and subtraction accumulate atomically in scatterNDCuda, and for any other op
  // a repeated destination, flagged on the device, hands the updates to scatterNDLockCuda.
  NDArray *marks = nullptr, *repeated = nullptr;
  if (!lock && op != pairwise::Add && op != pairwise::Subtract && rows > 1) {
    std::vector<LongType> marksShape = {(numDestinations + 31) / 32};
    marks = new NDArray('c', marksShape, INT32, context);
    repeated = new NDArray(INT32, context, true);
    marks->nullify();
    repeated->nullify();
    dim3 markDims = scatterNdDims(rows, indices.rankOf());

    NDArray::prepareSpecialUse({marks, repeated}, {&indices});
    BUILD_SINGLE_SELECTOR(xType, markRepeatedNdDestinationsCudaLauncher,
                          (markDims.x, markDims.y, context->getCudaStream(), indices.specialBuffer(),
                           indices.specialShapeInfo(), output.specialShapeInfo(), indexLength,
                           reinterpret_cast<unsigned int *>(marks->specialBuffer()),
                           reinterpret_cast<int *>(repeated->specialBuffer())),
                          SD_INDEXING_TYPES);
    NDArray::registerSpecialUse({marks, repeated}, {&indices});
  }

  dim3 perUpdateDims = scatterNdDims(updates.lengthOf(), updates.rankOf());
  dim3 orderedDims = scatterNdDims(sliceLen, updates.rankOf());
  const int *repeatedFlag = repeated != nullptr ? reinterpret_cast<const int *>(repeated->specialBuffer()) : nullptr;

  NDArray::prepareSpecialUse({&output}, {&updates, &indices, repeated});
  BUILD_DOUBLE_SELECTOR(xType, yType, scatterNDCudaLauncher,
                        (perUpdateDims, orderedDims, context->getCudaStream(), op, indices.specialBuffer(),
                         indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(),
                         output.specialBuffer(), output.specialShapeInfo(), indexLength, sliceLen, lock,
                         repeatedFlag),
                        SD_INDEXING_TYPES, SD_GENERIC_NUMERIC_TYPES);
  NDArray::registerSpecialUse({&output}, {&updates, &indices, repeated});

  manager.synchronize();
  delete marks;
  delete repeated;
  delete castUpdates;
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL SD_INLINE void scatterForLossCuda(const void* vx, const LongType* xShapeInfo, void* vy, const LongType* yShapeInfo,
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
BUILD_DOUBLE_TEMPLATE(void scatterForLossCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, void *vy, const LongType *yShapeInfo, void *vz, const LongType *zShapeInfo), SD_INDEXING_TYPES, SD_FLOAT_TYPES);

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
        (launchDIms.x, launchDIms.y, launchDIms.z, context->getCudaStream(), indices.specialBuffer(),
         indices.specialShapeInfo(), updates.specialBuffer(), updates.specialShapeInfo(), nullptr, nullptr),
        SD_INDEXING_TYPES, SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({&updates}, {&indices});
  } else {
    NDArray::prepareSpecialUse({&output}, {&indices, &updates});
    BUILD_DOUBLE_SELECTOR(indices.dataType(), updates.dataType(), scatterForLossCudaLauncher,
                          (launchDIms.x, launchDIms.y, launchDIms.z, context->getCudaStream(), indices.specialBuffer(),
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
